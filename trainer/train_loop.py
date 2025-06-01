"""
Цикл навчання моделі.
"""
import os
import json
import time
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Trainer:
    """Тренер моделі."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Ініціалізація.

        Args:
            model: Модель
            train_loader: Завантажувач тренувальних даних
            val_loader: Завантажувач валідаційних даних
            config: Конфігурація
            device: Пристрій
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or {}
        self.device = device

        # Оптимізатор
        self.optimizer = AdamW(
            model.parameters(),
            lr=self.config.get("learning_rate", 3e-4),
            weight_decay=self.config.get("weight_decay", 0.01)
        )

        # Планувальник
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config.get("warmup_steps", 1000),
            T_mult=1
        )

        # Логування
        self.writer = SummaryWriter(
            log_dir=self.config.get("log_dir", "runs")
        )

        # Метрики
        self.best_val_loss = float("inf")
        self.global_step = 0
        self.epoch = 0

    def train_epoch(self) -> float:
        """
        Навчання на одній епосі.

        Returns:
            Середня втрата
        """
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)

        with tqdm(self.train_loader, desc=f"Epoch {self.epoch}") as pbar:
            for batch in pbar:
                # Перенесення даних на пристрій
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                loss = nn.CrossEntropyLoss()(
                    outputs.view(-1, outputs.size(-1)),
                    labels.view(-1)
                )

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Градієнтний кліпінг
                if self.config.get("gradient_clip", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config["gradient_clip"]
                    )
                
                self.optimizer.step()
                self.scheduler.step()

                # Оновлення метрик
                total_loss += loss.item()
                self.global_step += 1

                # Логування
                if self.global_step % self.config.get("log_interval", 100) == 0:
                    self.writer.add_scalar(
                        "train/loss",
                        loss.item(),
                        self.global_step
                    )
                    self.writer.add_scalar(
                        "train/lr",
                        self.scheduler.get_last_lr()[0],
                        self.global_step
                    )

                # Оновлення прогресу
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
                })

                # Валідація
                if (
                    self.val_loader is not None
                    and self.global_step % self.config.get("validation_interval", 1000) == 0
                ):
                    val_loss = self.validate()
                    self.writer.add_scalar(
                        "val/loss",
                        val_loss,
                        self.global_step
                    )

                    # Збереження найкращої моделі
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint("model_best.pth")

                # Збереження чекпоінту
                if (
                    self.global_step % self.config.get("checkpoint_interval", 5000) == 0
                ):
                    self.save_checkpoint(f"checkpoint_{self.global_step}.pth")

        return total_loss / num_batches

    @torch.no_grad()
    def validate(self) -> float:
        """
        Валідація моделі.

        Returns:
            Середня втрата
        """
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)

        for batch in self.val_loader:
            # Перенесення даних на пристрій
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            # Forward pass
            outputs = self.model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(
                outputs.view(-1, outputs.size(-1)),
                labels.view(-1)
            )

            total_loss += loss.item()

        self.model.train()
        return total_loss / num_batches

    def save_checkpoint(self, path: str) -> None:
        """
        Збереження чекпоінту.

        Args:
            path: Шлях до файлу
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config
        }

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """
        Завантаження чекпоінту.

        Args:
            path: Шлях до файлу
        """
        checkpoint = torch.load(path)

        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        self.best_val_loss = checkpoint["best_val_loss"]
        self.config = checkpoint["config"]

    def train(self, num_epochs: int) -> None:
        """
        Навчання моделі.

        Args:
            num_epochs: Кількість епох
        """
        try:
            for epoch in range(self.epoch, num_epochs):
                self.epoch = epoch
                train_loss = self.train_epoch()

                # Логування епохи
                self.writer.add_scalar(
                    "epoch/train_loss",
                    train_loss,
                    epoch
                )

                # Збереження чекпоінту в кінці епохи
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pth")

        except KeyboardInterrupt:
            print("\nНавчання перервано користувачем")
            self.save_checkpoint("checkpoint_interrupted.pth")

        finally:
            self.writer.close() 