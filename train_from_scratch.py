"""
Script for training model from scratch using JSONL dataset.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import argparse
from tqdm import tqdm
import wandb
from datetime import datetime
import yaml
import random
import numpy as np
from transformers import get_linear_schedule_with_warmup
import os

from model.student_model import StudentModel
from tokenizer.bpe_pq_tokenizer import BPETokenizer
from utils.helpers import get_device, setup_logging
from tokenizer.simple_tokenizer import SimpleTokenizer
from model.transformer import TransformerModel
from data.dataset import (
    load_jsonl_dataset,
    split_dataset,
    create_dataloader,
    collate_fn
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class JSONLDataset(Dataset):
    """Dataset for JSONL files."""
    
    def __init__(self, file_path: str, tokenizer: BPETokenizer, max_length: int = 512):
        self.file_path = Path(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Load and preprocess data
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    example = json.loads(line.strip())
                    if "text" in example:
                        self.examples.append(example["text"])
                except json.JSONDecodeError:
                    continue
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.examples[idx]
        encoded = self.tokenizer.encode(text)
        
        # Truncate if needed
        if len(encoded["input_ids"]) > self.max_length:
            encoded["input_ids"] = encoded["input_ids"][:self.max_length]
            encoded["attention_mask"] = encoded["attention_mask"][:self.max_length]
        
        return {
            "input_ids": torch.tensor(encoded["input_ids"]),
            "attention_mask": torch.tensor(encoded["attention_mask"])
        }


class Trainer:
    """Trainer for model training."""
    
    def __init__(
        self,
        model: StudentModel,
        tokenizer: BPETokenizer,
        train_dataset: JSONLDataset,
        val_dataset: Optional[JSONLDataset],
        config: Dict,
        device: str,
        output_dir: str,
        use_wandb: bool = False
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.use_wandb = use_wandb
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Setup optimizer and scheduler
        self.optimizer = AdamW(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"]
        )
        
        # Calculate total steps
        self.total_steps = len(train_dataset) // config["training"]["batch_size"] * config["training"]["epochs"]
        
        # Setup scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config["training"]["warmup_steps"],
            num_training_steps=self.total_steps
        )
        
        # Setup mixed precision
        self.scaler = GradScaler() if config["training"]["use_fp16"] else None
        
        # Setup wandb
        if use_wandb:
            wandb.init(
                project="model-training",
                config=config,
                name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_tokens = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.config["training"]["use_fp16"]):
                outputs = self.model(input_ids, attention_mask)
                # Shift for next token prediction
                shift_logits = outputs[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                loss = nn.CrossEntropyLoss()(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["training"]["max_grad_norm"])
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["training"]["max_grad_norm"])
                self.optimizer.step()
            
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item()
            total_tokens += attention_mask.sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        return {
            "loss": total_loss / len(dataloader),
            "perplexity": torch.exp(torch.tensor(total_loss / len(dataloader))).item(),
            "tokens_per_second": total_tokens / (time.time() - progress_bar.start_t)
        }
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model."""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                # Forward pass
                with autocast(enabled=self.config["training"]["use_fp16"]):
                    outputs = self.model(input_ids, attention_mask)
                    shift_logits = outputs[..., :-1, :].contiguous()
                    shift_labels = input_ids[..., 1:].contiguous()
                    loss = nn.CrossEntropyLoss()(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                
                # Update metrics
                total_loss += loss.item()
                total_tokens += attention_mask.sum().item()
        
        return {
            "val_loss": total_loss / len(dataloader),
            "val_perplexity": torch.exp(torch.tensor(total_loss / len(dataloader))).item(),
            "val_tokens_per_second": total_tokens / (time.time() - progress_bar.start_t)
        }
    
    def train(self):
        """Train model."""
        # Create dataloaders
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
            num_workers=self.config["training"]["num_workers"],
            pin_memory=True
        )
        
        val_dataloader = None
        if self.val_dataset:
            val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=self.config["training"]["batch_size"],
                shuffle=False,
                num_workers=self.config["training"]["num_workers"],
                pin_memory=True
            )
        
        # Training loop
        best_val_loss = float("inf")
        for epoch in range(self.config["training"]["epochs"]):
            self.logger.info(f"Epoch {epoch + 1}/{self.config['training']['epochs']}")
            
            # Train
            train_metrics = self.train_epoch(train_dataloader)
            self.logger.info(f"Train metrics: {train_metrics}")
            
            # Evaluate
            if val_dataloader:
                val_metrics = self.evaluate(val_dataloader)
                self.logger.info(f"Validation metrics: {val_metrics}")
                
                # Save best model
                if val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    self.save_checkpoint(f"best_model.pt", epoch, train_metrics, val_metrics)
            
            # Save checkpoint
            self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt", epoch, train_metrics, val_metrics)
            
            # Log metrics
            if self.use_wandb:
                wandb.log({
                    **train_metrics,
                    **(val_metrics if val_dataloader else {}),
                    "epoch": epoch + 1
                })
    
    def save_checkpoint(self, filename: str, epoch: int, train_metrics: Dict[str, float], val_metrics: Optional[Dict[str, float]] = None):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "config": self.config
        }
        
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        torch.save(checkpoint, self.output_dir / filename)
        
        # Save tokenizer
        self.tokenizer.save(self.output_dir / "tokenizer.json")
        
        # Save config
        with open(self.output_dir / "config.yml", "w") as f:
            yaml.dump(self.config, f)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def train_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    epoch: int,
    gradient_accumulation_steps: int = 1
) -> float:
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch}",
        leave=False
    )

    for i, batch in enumerate(progress_bar):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["target_ids"]
        )
        
        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()
        
        if (i + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / len(train_loader)

def evaluate(
    model: torch.nn.Module,
    eval_loader: DataLoader,
    device: torch.device
) -> float:
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["target_ids"]
            )
            
            total_loss += outputs.loss.item()
    
    return total_loss / len(eval_loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to JSONL dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(True)
    logger = logging.getLogger(__name__)
    
    try:
        # Set random seeds
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        
        # Load config
        config = load_config(args.config)
        
        # Set device
        device = get_device(args.device)
        
        # Create tokenizer
        tokenizer = BPETokenizer()
        tokenizer.train(args.dataset, vocab_size=config["tokenizer"]["vocab_size"])
        
        # Create dataset
        dataset = JSONLDataset(args.dataset, tokenizer, config["student"]["max_seq_len"])
        
        # Split dataset
        val_size = int(len(dataset) * args.val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed)
        )
        
        # Create model
        model = TransformerModel(
            vocab_size=len(tokenizer.token_to_id),
            d_model=config["student"]["d_model"],
            n_heads=config["student"]["n_heads"],
            n_layers=config["student"]["n_layers"],
            max_seq_len=config["student"]["max_seq_len"],
            dropout=config["student"]["dropout"]
        ).to(device)
        
        # Create trainer
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=config,
            device=device,
            output_dir=args.output_dir,
            use_wandb=args.use_wandb
        )
        
        # Train model
        logger.info("Starting training...")
        trainer.train()
        logger.info("Training completed!")
    
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        raise
    finally:
        if args.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main() 