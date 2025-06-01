"""
Model trainer implementation.
"""
import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, List, Any
from tqdm import tqdm
import numpy as np
from model.transformer import TransformerModel
from tokenizer.simple_tokenizer import SimpleTokenizer
from utils.logger import setup_logger
import json


class Trainer:
    """Transformer trainer."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        max_grad_norm: float = 1.0,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        save_every: int = 1000,
        eval_every: int = 100,
        max_epochs: int = 10,
        early_stopping_patience: int = 3
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            tokenizer: Tokenizer
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            device: Training device
            learning_rate: Learning rate
            weight_decay: Weight decay
            warmup_steps: Number of warmup steps
            max_grad_norm: Maximum gradient norm
            checkpoint_dir: Checkpoint directory
            log_dir: Log directory
            save_every: Checkpoint save frequency
            eval_every: Validation frequency
            max_epochs: Maximum number of epochs
            early_stopping_patience: Early stopping patience
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.save_every = save_every
        self.eval_every = eval_every
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        
        self.logger = setup_logger(
            name="trainer",
            log_dir=log_dir,
            log_file="trainer.log"
        )
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max_epochs * len(train_dataloader),
            eta_min=learning_rate * 0.1
        )
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id["<pad>"])
        
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rates": []
        }

    def train_epoch(self) -> float:
        """
        Train for one epoch.

        Returns:
            Average epoch loss
        """
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_dataloader)
        
        pbar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.epoch + 1}/{self.max_epochs}",
            leave=False
        )
        
        for batch in pbar:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            logits = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                batch["target_ids"].view(-1)
            )
            
            self.optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm
            )
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            self.step += 1
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            if self.val_dataloader and self.step % self.eval_every == 0:
                val_loss = self.evaluate()
                self.logger.info(
                    f"Step {self.step}: val_loss = {val_loss:.4f}"
                )
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(epoch=self.epoch, loss=val_loss, is_best=True)
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
            
            if self.step % self.save_every == 0:
                self.save_checkpoint(epoch=self.epoch, loss=loss.item(), is_best=False)
        
        avg_loss = total_loss / num_batches
        self.history["train_loss"].append(avg_loss)
        self.history["learning_rates"].append(self.scheduler.get_last_lr()[0])
        
        return avg_loss

    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Validate model.

        Returns:
            Average validation loss
        """
        if not self.val_dataloader:
            return float("inf")
        
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_dataloader)
        
        for batch in self.val_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            logits = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                batch["target_ids"].view(-1)
            )
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        self.history["val_loss"].append(avg_loss)
        
        return avg_loss

    def train(self) -> Dict[str, Any]:
        """
        Train model.

        Returns:
            Training history
        """
        self.logger.info("Starting training")
        start_time = time.time()
        
        try:
            for epoch in range(self.max_epochs):
                self.epoch = epoch
                
                train_loss = self.train_epoch()
                
                val_loss = self.evaluate()
                
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.max_epochs}: "
                    f"train_loss = {train_loss:.4f}, "
                    f"val_loss = {val_loss:.4f}, "
                    f"lr = {self.scheduler.get_last_lr()[0]:.2e}"
                )
                
                if self.patience_counter >= self.early_stopping_patience:
                    self.logger.info(
                        f"Early stopping at epoch {epoch + 1} "
                        f"due to no improvement"
                    )
                    break
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        
        finally:
            training_time = time.time() - start_time
            self.logger.info(
                f"Training completed in {training_time:.2f} seconds"
            )
            
            return self.history

    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """
        Save checkpoint.

        Args:
            epoch: Current epoch
            loss: Current epoch loss
            is_best: Whether this is the best checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss
        }
        
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'checkpoint_epoch_{epoch}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint

        Returns:
            Epoch number and loss from checkpoint
        """
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['loss']

    def generate_text(
        self,
        prompt: str,
        max_length: int = 100,
        strategy: str = "greedy",
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        beam_size: int = 1,
        num_return_sequences: int = 1
    ) -> List[str]:
        """
        Generate text.

        Args:
            prompt: Input text
            max_length: Maximum length
            strategy: Generation strategy
            temperature: Sampling temperature
            top_k: Number of top tokens
            top_p: Nucleus sampling threshold
            beam_size: Beam size
            num_return_sequences: Number of sequences to return

        Returns:
            List of generated texts
        """
        self.model.eval()
        
        with torch.no_grad():
            return self.model.generate(
                prompt=prompt,
                tokenizer=self.tokenizer,
                max_len=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                beam_size=beam_size
            ) 