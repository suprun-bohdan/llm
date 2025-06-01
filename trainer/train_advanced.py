"""
Advanced training script with hypernetwork, distillation, pruning and quantization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union

from hypernetwork.hypernet import HyperNetwork
from model.student_model import StudentModel
from model.distillation import DistillationTrainer
from pruning_quant.pruning import MagnitudePruner, FisherPruner
from pruning_quant.quantization import DynamicQuantizer, StaticQuantizer
from rag_memory.memory_bank import MemoryBank
from tokenizer.bpe_pq_tokenizer import BPETokenizer
from utils.helpers import get_device, count_parameters
from utils.logger import Logger
from data.dataset import load_jsonl_dataset, TextDataset, create_dataloader


def load_config(config_path: str) -> Dict:
    """
    Load configuration.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_model(config: Dict, device: str) -> Tuple[nn.Module, nn.Module]:
    """
    Create teacher and student models.

    Args:
        config: Configuration dictionary
        device: Device to use

    Returns:
        Teacher and student models
    """
    # Create teacher model
    teacher = StudentModel(
        vocab_size=config["tokenizer"]["vocab_size"],
        d_model=config["student"]["d_model"],
        n_heads=config["student"]["n_heads"],
        n_layers=config["student"]["n_layers"],
        d_ff=config["student"]["d_model"] * 4,
        max_seq_len=config["student"]["max_seq_len"],
        dropout=config["student"]["dropout"],
        lora_rank=config["student"]["lora_rank"],
        distill_alpha=config["student"]["distill_alpha"],
        use_lora=False,
        ffn_rank=config["student"]["ffn_rank"],
        gradient_checkpointing=config["student"]["gradient_checkpointing"],
        mixed_precision=config["student"]["mixed_precision"]
    )
    
    # Load teacher weights if available
    if "teacher_path" in config:
        teacher.load_state_dict(torch.load(config["teacher_path"]))
    
    teacher.to(device)
    teacher.eval()
    
    # Create student model
    student = StudentModel(
        vocab_size=config["tokenizer"]["vocab_size"],
        d_model=config["student"]["d_model"],
        n_heads=config["student"]["n_heads"],
        n_layers=config["student"]["n_layers"],
        d_ff=config["student"]["d_model"] * 4,
        max_seq_len=config["student"]["max_seq_len"],
        dropout=config["student"]["dropout"],
        lora_rank=config["student"]["lora_rank"],
        distill_alpha=config["student"]["distill_alpha"],
        use_lora=config["student"]["use_lora"],
        ffn_rank=config["student"]["ffn_rank"],
        gradient_checkpointing=config["student"]["gradient_checkpointing"],
        mixed_precision=config["student"]["mixed_precision"]
    )
    
    student.to(device)
    
    return teacher, student


def create_hypernetwork(config: Dict, device: str) -> Optional[HyperNetwork]:
    """
    Create hypernetwork if enabled.

    Args:
        config: Configuration dictionary
        device: Device to use

    Returns:
        Hypernetwork or None
    """
    if not config["use_hypernet"]:
        return None
    
    # Get layer shapes
    layer_shapes = []
    for i in range(config["student"]["n_layers"]):
        # Attention layers
        layer_shapes.extend([
            (config["student"]["d_model"], config["student"]["d_model"]),  # Q
            (config["student"]["d_model"], config["student"]["d_model"]),  # K
            (config["student"]["d_model"], config["student"]["d_model"]),  # V
            (config["student"]["d_model"], config["student"]["d_model"])   # O
        ])
        
        # FFN layers
        layer_shapes.extend([
            (config["student"]["d_model"], config["student"]["d_model"] * 4),  # W1
            (config["student"]["d_model"] * 4, config["student"]["d_model"])   # W2
        ])
    
    # Create hypernetwork
    hypernet = HyperNetwork(
        hidden_sizes=config["hypernet"]["hidden_sizes"],
        layer_shapes=layer_shapes,
        activation=config["hypernet"]["activation"]
    )
    
    hypernet.to(device)
    return hypernet


def create_optimizer(
    model: nn.Module,
    hypernet: Optional[HyperNetwork],
    config: Dict
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """
    Create optimizer and scheduler.

    Args:
        model: Model to optimize
        hypernet: Hypernetwork (optional)
        config: Configuration dictionary

    Returns:
        Optimizer and scheduler
    """
    # Get parameters
    params = list(model.parameters())
    if hypernet is not None:
        params.extend(hypernet.parameters())
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        params,
        lr=config["optimizer"]["learning_rate"],
        weight_decay=config["optimizer"]["weight_decay"]
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config["optimizer"]["warmup_steps"],
        T_mult=2
    )
    
    return optimizer, scheduler


def train_epoch(
    teacher: nn.Module,
    student: nn.Module,
    hypernet: Optional[HyperNetwork],
    distill_trainer: DistillationTrainer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    config: Dict,
    logger: Logger,
    epoch: int,
    device: str
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        teacher: Teacher model
        student: Student model
        hypernet: Hypernetwork (optional)
        distill_trainer: Distillation trainer
        dataloader: Training dataloader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        scaler: Gradient scaler
        config: Configuration dictionary
        logger: Logger
        epoch: Current epoch
        device: Device to use

    Returns:
        Dictionary with metrics
    """
    student.train()
    total_loss = 0
    total_tokens = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for i, batch in enumerate(progress_bar):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Generate weights if using hypernetwork
        if hypernet is not None:
            weights = hypernet()
            student.load_weights(weights)
        
        # Forward pass with mixed precision
        with autocast():
            loss, loss_components = distill_trainer.train_step(
                batch,
                optimizer
            )
        
        # Scale loss and backward pass
        scaler.scale(loss).backward()
        
        # Update weights
        if (i + 1) % config["gradient_accumulation_steps"] == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
        
        # Update metrics
        total_loss += loss.item() * batch["input_ids"].size(0)
        total_tokens += batch["attention_mask"].sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            "loss": loss.item(),
            "lr": scheduler.get_last_lr()[0]
        })
        
        # Log metrics
        if (i + 1) % config["logging_steps"] == 0:
            logger.log_metrics({
                "train/loss": loss.item(),
                "train/lr": scheduler.get_last_lr()[0],
                **{f"train/{k}": v for k, v in loss_components.items()}
            }, epoch * len(dataloader) + i)
    
    return {
        "train_loss": total_loss / len(dataloader),
        "train_perplexity": torch.exp(torch.tensor(total_loss / total_tokens)).item()
    }


def evaluate(
    teacher: nn.Module,
    student: nn.Module,
    hypernet: Optional[HyperNetwork],
    distill_trainer: DistillationTrainer,
    dataloader: DataLoader,
    config: Dict,
    logger: Logger,
    epoch: int,
    device: str
) -> Dict[str, float]:
    """
    Evaluate model.

    Args:
        teacher: Teacher model
        student: Student model
        hypernet: Hypernetwork (optional)
        distill_trainer: Distillation trainer
        dataloader: Evaluation dataloader
        config: Configuration dictionary
        logger: Logger
        epoch: Current epoch
        device: Device to use

    Returns:
        Dictionary with metrics
    """
    metrics = distill_trainer.evaluate(dataloader)
    
    # Log metrics
    logger.log_metrics({
        f"eval/{k}": v for k, v in metrics.items()
    }, epoch)
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--pretrain_hypernet", action="store_true", help="Pretrain hypernetwork")
    parser.add_argument("--train_student", action="store_true", help="Train student model")
    parser.add_argument("--apply_pruning", action="store_true", help="Apply pruning after training")
    parser.add_argument("--quantize", action="store_true", help="Apply quantization after training")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = get_device()
    
    # Create logger
    logger = Logger(output_dir / "logs", "advanced_training")
    logger.log_config(config)
    
    # Create models
    teacher, student = create_model(config, device)
    logger.log_model_graph(student)
    
    # Create hypernetwork
    hypernet = create_hypernetwork(config, device)
    
    # Pretrain hypernetwork if requested
    if args.pretrain_hypernet and hypernet is not None:
        print("Pretraining hypernetwork...")
        hypernet_optimizer = torch.optim.AdamW(
            hypernet.parameters(),
            lr=config["hypernet"]["learning_rate"]
        )
        
        # Pretrain hypernetwork
        for epoch in range(config["hypernet"]["pretrain_epochs"]):
            hypernet.train()
            total_loss = 0
            
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Generate weights
                weights = hypernet()
                
                # Load weights into student
                student.load_weights(weights)
                
                # Forward pass
                outputs = student(batch["input_ids"], batch["attention_mask"])
                loss = F.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    batch["target_ids"].view(-1)
                )
                
                # Backward pass
                hypernet_optimizer.zero_grad()
                loss.backward()
                hypernet_optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            logger.log_metrics({
                "hypernet/pretrain_loss": avg_loss
            }, epoch)
            
            print(f"Hypernetwork pretrain epoch {epoch + 1}: loss = {avg_loss:.4f}")
        
        # Save pretrained hypernetwork
        torch.save(
            hypernet.state_dict(),
            output_dir / "hypernet_pretrained.pt"
        )
    
    # Train student model if requested
    if args.train_student:
        # Create optimizer
        optimizer, scheduler = create_optimizer(student, hypernet, config)
        
        # Create distillation trainer
        distill_trainer = DistillationTrainer(
            student=student,
            teacher=teacher,
            alpha=config["student"]["distill_alpha"],
            temperature=config["student"]["temperature"],
            device=device
        )
        
        # Create gradient scaler
        scaler = GradScaler()
        
        # Load dataset
        train_texts = load_jsonl_dataset(config["train_dataset"])
        eval_texts = load_jsonl_dataset(config["eval_dataset"])

        # Create datasets
        train_dataset = TextDataset(
            texts=train_texts,
            tokenizer=tokenizer,
            max_length=config["tokenizer"]["max_seq_len"]
        )
        eval_dataset = TextDataset(
            texts=eval_texts,
            tokenizer=tokenizer,
            max_length=config["tokenizer"]["max_seq_len"]
        )

        # Create dataloaders
        train_loader = create_dataloader(
            texts=train_texts,
            tokenizer=tokenizer,
            max_seq_len=config["tokenizer"]["max_seq_len"],
            batch_size=config["batch_size"],
            shuffle=True
        )

        eval_loader = create_dataloader(
            texts=eval_texts,
            tokenizer=tokenizer,
            max_seq_len=config["tokenizer"]["max_seq_len"],
            batch_size=config["batch_size"],
            shuffle=False
        )
        
        # Training loop
        best_eval_loss = float("inf")
        
        for epoch in range(config["epochs"]):
            # Train
            train_metrics = train_epoch(
                teacher,
                student,
                hypernet,
                distill_trainer,
                train_loader,
                optimizer,
                scheduler,
                scaler,
                config,
                logger,
                epoch,
                device
            )
            
            # Evaluate
            eval_metrics = evaluate(
                teacher,
                student,
                hypernet,
                distill_trainer,
                eval_loader,
                config,
                logger,
                epoch,
                device
            )
            
            # Save checkpoint
            if eval_metrics["eval_loss"] < best_eval_loss:
                best_eval_loss = eval_metrics["eval_loss"]
                distill_trainer.save_checkpoint(
                    output_dir / "best_model.pt",
                    optimizer,
                    epoch,
                    {**train_metrics, **eval_metrics}
                )
            
            # Apply pruning if requested
            if args.apply_pruning and epoch == config["pruning"]["start_epoch"]:
                print("Applying pruning...")
                if config["pruning"]["method"] == "magnitude":
                    pruner = MagnitudePruner(
                        student,
                        amount=config["pruning"]["prune_rate"],
                        schedule=config["pruning"]["schedule"],
                        start_epoch=config["pruning"]["start_epoch"],
                        end_epoch=config["pruning"]["end_epoch"]
                    )
                else:
                    pruner = FisherPruner(
                        student,
                        amount=config["pruning"]["prune_rate"],
                        n_samples=config["pruning"]["n_samples"],
                        device=device
                    )
                    pruner.prune(eval_loader)
        
        # Apply quantization if requested
        if args.quantize:
            print("Applying quantization...")
            if config["quantization"]["method"] == "dynamic":
                quantizer = DynamicQuantizer(
                    bits=config["quantization"]["bits"],
                    symmetric=config["quantization"]["symmetric"],
                    per_channel=config["quantization"]["per_channel"]
                )
            else:
                quantizer = StaticQuantizer(
                    bits=config["quantization"]["bits"],
                    symmetric=config["quantization"]["symmetric"],
                    per_channel=config["quantization"]["per_channel"],
                    calibration_data=eval_loader
                )
                quantizer.calibrate(student)
            
            student = quantizer.quantize_model(student)
            student.to(device)
            
            # Save quantized model
            torch.save(
                student.state_dict(),
                output_dir / "model_quant.pt"
            )
    
    # Build memory bank if enabled
    if config["rag"]["enabled"]:
        print("Building memory bank...")
        memory_bank = MemoryBank(
            dim=config["rag"]["dim"],
            nlist=config["rag"]["nlist"],
            m=config["rag"]["m"],
            nbits=config["rag"]["nbits"],
            device=device
        )
        
        # Get embeddings
        embeddings = []
        metadata = []
        
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = student(batch["input_ids"], batch["attention_mask"])
                batch_emb = outputs.mean(dim=1)
                embeddings.append(batch_emb.cpu())
                
                for text in batch["text"]:
                    metadata.append({"text": text})
        
        embeddings = torch.cat(embeddings, dim=0)
        
        # Build index
        memory_bank.build(embeddings, metadata)
        memory_bank.save(
            output_dir / "memory_bank.faiss",
            output_dir / "memory_bank.json"
        )
    
    logger.close()


if __name__ == "__main__":
    main() 