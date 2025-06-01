"""
Script for model training.
"""
import os
import argparse
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from model.transformer import TransformerModel
from tokenizer.simple_tokenizer import SimpleTokenizer
from trainer.trainer import Trainer
from data.dataset import (
    create_dataloader,
    load_jsonl_dataset,
    split_dataset
)
from utils.logger import setup_logger
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb
from student.model_student import StudentModel
import json


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Transformer training"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data file (JSONL)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory for saving results"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint for resuming training"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducibility"
    )
    
    parser.add_argument(
        "--teacher_model_path",
        type=str,
        help="Path to teacher model checkpoint"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """
    Load configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def setup_directories(output_dir: str) -> tuple:
    """
    Create directories for saving results.

    Args:
        output_dir: Base directory

    Returns:
        Tuple with directory paths
    """
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    log_dir = os.path.join(output_dir, "logs")
    model_dir = os.path.join(output_dir, "model")
    
    for directory in [checkpoint_dir, log_dir, model_dir]:
        os.makedirs(directory, exist_ok=True)
    
    return checkpoint_dir, log_dir, model_dir


class TextDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=256):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip()
                if text:
                    self.data.append(text)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze()
        }


def train_epoch(model, dataloader, optimizer, scheduler, device, distill_alpha=0.5):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids, attention_mask)
        
        # Обчислюємо втрати для мови моделювання
        shift_logits = outputs[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss()
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Якщо є вчительська модель, додаємо дистиляційні втрати
        if hasattr(model, 'teacher_model') and model.teacher_model is not None:
            with torch.no_grad():
                teacher_outputs = model.teacher_model(input_ids, attention_mask)
            
            distill_loss = nn.KLDivLoss()(
                F.log_softmax(shift_logits / model.distill_temperature, dim=-1),
                F.softmax(teacher_outputs[..., :-1, :].contiguous() / model.distill_temperature, dim=-1)
            ) * (model.distill_temperature ** 2)
            
            loss = distill_alpha * distill_loss + (1 - distill_alpha) * lm_loss
        else:
            loss = lm_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
        
        wandb.log({
            'train/loss': loss.item(),
            'train/lm_loss': lm_loss.item() if isinstance(lm_loss, torch.Tensor) else lm_loss,
            'train/distill_loss': distill_loss.item() if 'distill_loss' in locals() else 0,
            'train/learning_rate': scheduler.get_last_lr()[0]
        })
    
    return total_loss / len(dataloader)


def main():
    """Main function."""
    args = parse_args()
    
    logger = setup_logger(
        name="train",
        log_dir=os.path.join(args.output_dir, "logs"),
        log_file="train.log"
    )
    
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    checkpoint_dir, log_dir, model_dir = setup_directories(args.output_dir)
    logger.info(f"Created directories in {args.output_dir}")
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    logger.info(f"Set seed: {args.seed}")
    
    texts = load_jsonl_dataset(
        file_path=args.data,
        text_field=config["data"]["text_field"],
        max_samples=config["data"].get("max_samples")
    )
    logger.info(f"Loaded {len(texts)} texts")
    
    train_texts, val_texts, _ = split_dataset(
        texts=texts,
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
        test_ratio=config["data"]["test_ratio"],
        seed=args.seed
    )
    logger.info(
        f"Split datasets: "
        f"training ({len(train_texts)}), "
        f"validation ({len(val_texts)})"
    )
    
    tokenizer = SimpleTokenizer(
        vocab_size=config["tokenizer"]["vocab_size"],
        min_freq=config["tokenizer"]["min_freq"],
        special_tokens=config["tokenizer"]["special_tokens"]
    )
    
    tokenizer.train(train_texts)
    logger.info(
        f"Trained tokenizer with vocabulary size "
        f"{len(tokenizer.token_to_id)}"
    )
    
    train_dataloader = create_dataloader(
        texts=train_texts,
        tokenizer=tokenizer,
        max_seq_len=config["model"]["max_seq_len"],
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pad_token_id=tokenizer.token_to_id["<pad>"],
        bos_token_id=tokenizer.token_to_id["<bos>"],
        eos_token_id=tokenizer.token_to_id["<eos>"]
    )
    
    val_dataloader = create_dataloader(
        texts=val_texts,
        tokenizer=tokenizer,
        max_seq_len=config["model"]["max_seq_len"],
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pad_token_id=tokenizer.token_to_id["<pad>"],
        bos_token_id=tokenizer.token_to_id["<bos>"],
        eos_token_id=tokenizer.token_to_id["<eos>"]
    )
    logger.info("Created data loaders")
    
    model = StudentModel(
        vocab_size=len(tokenizer.token_to_id),
        d_model=config["model"]["d_model"],
        n_layers=config["model"]["n_layers"],
        n_heads=config["model"]["n_heads"],
        d_ff=config["model"]["d_ff"],
        dropout=config["model"]["dropout"],
        lora_rank=config["model"]["lora_rank"],
        distill_alpha=config["model"]["distill_alpha"],
        use_lora=config["model"]["use_lora"],
        ffn_rank=config["model"]["ffn_rank"],
        gradient_checkpointing=config["model"]["gradient_checkpointing"],
        mixed_precision=config["model"]["mixed_precision"],
        max_seq_len=config["model"]["max_seq_len"],
        activation=config["model"]["activation"]
    )
    logger.info("Created model")
    
    if args.teacher_model_path:
        teacher_model = torch.load(args.teacher_model_path).to(config["training"]["device"])
        teacher_model.eval()
        model.teacher_model = teacher_model
        model.distill_temperature = config["model"]["distill_temperature"]
    
    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["max_epochs"] * len(train_dataloader),
        eta_min=config["training"]["min_learning_rate"]
    )
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=config["training"]["device"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        warmup_steps=config["training"]["warmup_steps"],
        max_grad_norm=config["training"]["max_grad_norm"],
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        save_every=config["training"]["save_every"],
        eval_every=config["training"]["eval_every"],
        max_epochs=config["training"]["max_epochs"],
        early_stopping_patience=config["training"]["early_stopping_patience"]
    )
    logger.info("Created trainer")
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed training from {args.resume}")
    
    wandb.init(
        project="student-model-training",
        config=config
    )
    
    best_loss = float('inf')
    for epoch in range(config["training"]["max_epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['training']['max_epochs']}")
        
        train_loss = train_epoch(
            model,
            train_dataloader,
            optimizer,
            scheduler,
            config["training"]["device"],
            config["model"]["distill_alpha"]
        )
        
        print(f"Average training loss: {train_loss:.4f}")
        
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pt'))
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': train_loss,
        }, os.path.join(checkpoint_dir, 'checkpoint.pt'))
        
        wandb.log({
            'epoch': epoch,
            'train/avg_loss': train_loss
        })
    
    wandb.finish()
    
    model_path = os.path.join(model_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved model to {model_path}")
    
    tokenizer_path = os.path.join(model_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    logger.info(f"Saved tokenizer to {tokenizer_path}")
    
    config_path = os.path.join(model_dir, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f)
    logger.info(f"Saved configuration to {config_path}")


if __name__ == "__main__":
    main() 