"""
Реалізація тренера моделі.
"""
import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, List
from tqdm import tqdm
import numpy as np
from model.transformer import TransformerModel
from tokenizer.simple_tokenizer import SimpleTokenizer
from utils.logger import setup_logger


class Trainer:
    """Тренер для навчання трансформера."""

    def __init__(
        self,
        model: TransformerModel,
        tokenizer: SimpleTokenizer,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_grad_norm: float = 1.0,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        save_every: int = 1000,
        eval_every: int = 100,
        max_epochs: int = 10,
        early_stopping_patience: int = 3
    ):
        """
        Ініціалізація тренера.

        Args:
            model: Модель для навчання
            tokenizer: Токенізатор
            train_dataloader: Завантажувач тренувальних даних
            val_dataloader: Завантажувач валідаційних даних
            device: Пристрій для навчання
            learning_rate: Швидкість навчання
            weight_decay: Ваговий розпад
            warmup_steps: Кількість кроків для розігріву
            max_grad_norm: Максимальна норма градієнта
            checkpoint_dir: Директорія для збереження чекпоінтів
            log_dir: Директорія для логів
            save_every: Частота збереження чекпоінтів
            eval_every: Частота валідації
            max_epochs: Максимальна кількість епох
            early_stopping_patience: Терпіння для раннього зупинки
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        
        # Оптимізатор
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Планувальник швидкості навчання
        total_steps = len(train_dataloader) * max_epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=learning_rate * 0.1
        )
        
        # Критерій втрат
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id["<pad>"])
        
        # Параметри навчання
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.save_every = save_every
        self.eval_every = eval_every
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        
        # Створення директорій
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Логування
        self.logger = setup_logger(
            name="trainer",
            log_dir=log_dir,
            log_file="train.log"
        )
        
        # Метрики
        self.best_val_loss = float("inf")
        self.epoch = 0
        self.step = 0
        self.patience_counter = 0
        
        # Історія навчання
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rates": []
        }

    def train_epoch(self) -> float:
        """
        Навчання на одній епосі.

        Returns:
            Середня втрата на епосі
        """
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_dataloader)
        
        # Прогрес-бар
        pbar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.epoch + 1}/{self.max_epochs}",
            leave=False
        )
        
        for batch in pbar:
            # Перенесення батчу на пристрій
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Прямий прохід
            logits = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            
            # Розрахунок втрат
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                batch["target_ids"].view(-1)
            )
            
            # Зворотній прохід
            self.optimizer.zero_grad()
            loss.backward()
            
            # Обрізання градієнтів
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm
            )
            
            # Оновлення параметрів
            self.optimizer.step()
            self.scheduler.step()
            
            # Оновлення метрик
            total_loss += loss.item()
            self.step += 1
            
            # Оновлення прогрес-бару
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Валідація
            if self.val_dataloader and self.step % self.eval_every == 0:
                val_loss = self.evaluate()
                self.logger.info(
                    f"Step {self.step}: val_loss = {val_loss:.4f}"
                )
                
                # Збереження найкращої моделі
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best_model.pt")
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
            
            # Збереження чекпоінту
            if self.step % self.save_every == 0:
                self.save_checkpoint(f"checkpoint_{self.step}.pt")
        
        # Розрахунок середньої втрати
        avg_loss = total_loss / num_batches
        self.history["train_loss"].append(avg_loss)
        self.history["learning_rates"].append(self.scheduler.get_last_lr()[0])
        
        return avg_loss

    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Валідація моделі.

        Returns:
            Середня втрата на валідаційному наборі
        """
        if not self.val_dataloader:
            return float("inf")
        
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_dataloader)
        
        for batch in self.val_dataloader:
            # Перенесення батчу на пристрій
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Прямий прохід
            logits = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            
            # Розрахунок втрат
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                batch["target_ids"].view(-1)
            )
            
            total_loss += loss.item()
        
        # Розрахунок середньої втрати
        avg_loss = total_loss / num_batches
        self.history["val_loss"].append(avg_loss)
        
        return avg_loss

    def train(self) -> Dict[str, List[float]]:
        """
        Навчання моделі.

        Returns:
            Історія навчання
        """
        self.logger.info("Початок навчання")
        start_time = time.time()
        
        try:
            for epoch in range(self.max_epochs):
                self.epoch = epoch
                
                # Навчання на епосі
                train_loss = self.train_epoch()
                
                # Валідація
                val_loss = self.evaluate()
                
                # Логування
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.max_epochs}: "
                    f"train_loss = {train_loss:.4f}, "
                    f"val_loss = {val_loss:.4f}, "
                    f"lr = {self.scheduler.get_last_lr()[0]:.2e}"
                )
                
                # Перевірка раннього зупинки
                if self.patience_counter >= self.early_stopping_patience:
                    self.logger.info(
                        f"Раннє зупинка на епосі {epoch + 1} "
                        f"через відсутність покращення"
                    )
                    break
        
        except KeyboardInterrupt:
            self.logger.info("Навчання перервано користувачем")
        
        finally:
            # Збереження фінальної моделі
            self.save_checkpoint("final_model.pt")
            
            # Логування часу навчання
            training_time = time.time() - start_time
            self.logger.info(
                f"Навчання завершено за {training_time:.2f} секунд"
            )
        
        return self.history

    def save_checkpoint(self, filename: str) -> None:
        """
        Збереження чекпоінту.

        Args:
            filename: Назва файлу
        """
        path = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "epoch": self.epoch,
            "step": self.step,
            "best_val_loss": self.best_val_loss,
            "history": self.history
        }
        
        torch.save(checkpoint, path)
        self.logger.info(f"Збережено чекпоінт: {path}")

    def load_checkpoint(self, filename: str) -> None:
        """
        Завантаження чекпоінту.

        Args:
            filename: Назва файлу
        """
        path = os.path.join(self.checkpoint_dir, filename)
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Чекпоінт не знайдено: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        
        self.epoch = checkpoint["epoch"]
        self.step = checkpoint["step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.history = checkpoint["history"]
        
        self.logger.info(f"Завантажено чекпоінт: {path}")

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
        Генерація тексту.

        Args:
            prompt: Початковий текст
            max_length: Максимальна довжина
            strategy: Стратегія вибірки
            temperature: Температура
            top_k: Кількість найкращих токенів
            top_p: Поріг для nucleus sampling
            beam_size: Розмір променя
            num_return_sequences: Кількість послідовностей

        Returns:
            Список згенерованих текстів
        """
        self.model.eval()
        
        # Токенізація промпту
        input_ids = torch.tensor(
            [self.tokenizer.encode(prompt)],
            device=self.device
        )
        
        # Генерація
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                max_length=max_length,
                strategy=strategy,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                beam_size=beam_size,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.token_to_id["<pad>"],
                eos_token_id=self.tokenizer.token_to_id["<eos>"]
            )
        
        # Детокенізація
        texts = []
        for ids in output_ids:
            text = self.tokenizer.decode(ids.tolist())
            texts.append(text)
        
        return texts 