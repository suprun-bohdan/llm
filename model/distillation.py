"""
Дистиляція моделі.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Any
import math
from model.student_model import StudentModel


class DistillationLoss(nn.Module):
    """Функція втрат для дистиляції."""
    
    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,
        reduction: str = "mean"
    ):
        """
        Ініціалізація функції втрат.

        Args:
            temperature: Температура для softmax
            alpha: Вага для дистиляції
            reduction: Тип редукції
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.reduction = reduction
        
        self.kl_div = nn.KLDivLoss(reduction=reduction)
        self.ce = nn.CrossEntropyLoss(reduction=reduction)
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Обчислення втрат.

        Args:
            student_logits: Логіти студента
            teacher_logits: Логіти вчителя
            labels: Мітки

        Returns:
            Втрати
        """
        student_log_probs = F.log_softmax(
            student_logits / self.temperature,
            dim=-1
        )
        teacher_probs = F.softmax(
            teacher_logits / self.temperature,
            dim=-1
        )
        distillation_loss = self.kl_div(
            student_log_probs,
            teacher_probs
        ) * (self.temperature ** 2)
        
        student_loss = self.ce(student_logits, labels)
        
        return (
            self.alpha * distillation_loss +
            (1 - self.alpha) * student_loss
        )


class DistillationTrainer:
    """Тренер для дистиляції."""
    
    def __init__(
        self,
        student: StudentModel,
        teacher: nn.Module,
        optimizer: torch.optim.Optimizer,
        temperature: float = 2.0,
        alpha: float = 0.5,
        device: str = "cuda"
    ):
        """
        Ініціалізація тренера.

        Args:
            student: Студент
            teacher: Вчитель
            optimizer: Оптимізатор
            temperature: Температура
            alpha: Вага дистиляції
            device: Пристрій
        """
        self.student = student
        self.teacher = teacher
        self.optimizer = optimizer
        self.device = device
        
        self.criterion = DistillationLoss(
            temperature=temperature,
            alpha=alpha
        )
        
        self.student.to(device)
        self.teacher.to(device)
        
        self.teacher.eval()
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Крок навчання.

        Args:
            batch: Пакет даних

        Returns:
            Словник з метриками
        """
        self.student.train()
        self.optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["target_ids"].to(self.device)
        
        student_logits = self.student(input_ids, attention_mask)
        
        with torch.no_grad():
            teacher_input = self.student.embed_tokens(input_ids) + self.student.pos_embed(
                torch.arange(0, input_ids.size(1), device=input_ids.device).unsqueeze(0).repeat(input_ids.size(0), 1)
            )
            teacher_logits = self.teacher(teacher_input, src_key_padding_mask=attention_mask == 0)
            teacher_logits = self.student.output_proj(teacher_logits)
        
        loss = self.criterion(
            student_logits,
            teacher_logits,
            labels
        )
        
        loss.backward()
        self.optimizer.step()
        
        with torch.no_grad():
            student_probs = F.softmax(student_logits, dim=-1)
            teacher_probs = F.softmax(teacher_logits, dim=-1)
            
            student_preds = student_probs.argmax(dim=-1)
            accuracy = (student_preds == labels).float().mean()
            
            kl_div = F.kl_div(
                student_probs.log(),
                teacher_probs,
                reduction="batchmean"
            )
        
        return {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "kl_div": kl_div.item()
        }
    
    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """
        Оцінка моделі.

        Args:
            dataloader: Завантажувач даних

        Returns:
            Словник з метриками
        """
        self.student.eval()
        total_loss = 0
        total_accuracy = 0
        total_kl_div = 0
        total_tokens = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["target_ids"].to(self.device)
                
                student_logits = self.student(input_ids, attention_mask)
                
                teacher_input = self.student.embed_tokens(input_ids) + self.student.pos_embed(
                    torch.arange(0, input_ids.size(1), device=input_ids.device).unsqueeze(0).repeat(input_ids.size(0), 1)
                )
                teacher_logits = self.teacher(teacher_input, src_key_padding_mask=attention_mask == 0)
                teacher_logits = self.student.output_proj(teacher_logits)
                
                loss = self.criterion(
                    student_logits,
                    teacher_logits,
                    labels
                )
                
                student_probs = F.softmax(student_logits, dim=-1)
                teacher_probs = F.softmax(teacher_logits, dim=-1)
                
                student_preds = student_probs.argmax(dim=-1)
                accuracy = (student_preds == labels).float().mean()
                
                kl_div = F.kl_div(
                    student_probs.log(),
                    teacher_probs,
                    reduction="batchmean"
                )
                
                total_loss += loss.item()
                total_accuracy += accuracy.item()
                total_kl_div += kl_div.item()
                total_tokens += attention_mask.sum().item()
                num_batches += 1
        
        return {
            "eval_loss": total_loss / num_batches,
            "eval_accuracy": total_accuracy / num_batches,
            "eval_kl_div": total_kl_div / num_batches,
            "eval_perplexity": torch.exp(torch.tensor(total_loss / total_tokens)).item()
        }
    
    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        metrics: Dict[str, float]
    ):
        """
        Збереження чекпоінту.

        Args:
            path: Шлях для збереження
            epoch: Номер епохи
            metrics: Метрики
        """
        torch.save({
            "student_state": self.student.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics
        }, path)
    
    def load_checkpoint(
        self,
        path: str
    ) -> Tuple[int, Dict[str, float]]:
        """
        Завантаження чекпоінту.

        Args:
            path: Шлях для завантаження

        Returns:
            Номер епохи та метрики
        """
        checkpoint = torch.load(path)
        self.student.load_state_dict(checkpoint["student_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        return checkpoint["epoch"], checkpoint["metrics"]


class ProgressiveDistillation:
    """Прогресивна дистиляція."""
    
    def __init__(
        self,
        student: StudentModel,
        teacher: nn.Module,
        optimizer: torch.optim.Optimizer,
        num_steps: int = 3,
        initial_temperature: float = 4.0,
        final_temperature: float = 1.0,
        initial_alpha: float = 0.9,
        final_alpha: float = 0.5,
        device: str = "cuda"
    ):
        """
        Ініціалізація прогресивної дистиляції.

        Args:
            student: Студент
            teacher: Вчитель
            optimizer: Оптимізатор
            num_steps: Кількість кроків
            initial_temperature: Початкова температура
            final_temperature: Кінцева температура
            initial_alpha: Початкова вага дистиляції
            final_alpha: Кінцева вага дистиляції
            device: Пристрій
        """
        self.student = student
        self.teacher = teacher
        self.optimizer = optimizer
        self.num_steps = num_steps
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.initial_alpha = initial_alpha
        self.final_alpha = final_alpha
        self.device = device
        
        self.student.to(device)
        self.teacher.to(device)
        
        self.teacher.eval()
    
    def get_step_params(self, step: int) -> Tuple[float, float]:
        """
        Отримання параметрів для кроку.

        Args:
            step: Номер кроку

        Returns:
            Кортеж (температура, вага)
        """
        progress = step / (self.num_steps - 1)
        temperature = (
            self.initial_temperature * (1 - progress) +
            self.final_temperature * progress
        )
        alpha = (
            self.initial_alpha * (1 - progress) +
            self.final_alpha * progress
        )
        return temperature, alpha
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        step: int
    ) -> Dict[str, float]:
        """
        Крок навчання.

        Args:
            batch: Пакет даних
            step: Номер кроку

        Returns:
            Словник з метриками
        """
        temperature, alpha = self.get_step_params(step)
        criterion = DistillationLoss(
            temperature=temperature,
            alpha=alpha
        )
        
        self.student.train()
        self.optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["target_ids"].to(self.device)
        
        student_logits = self.student(input_ids, attention_mask)
        
        with torch.no_grad():
            teacher_input = self.student.embed_tokens(input_ids) + self.student.pos_embed(
                torch.arange(0, input_ids.size(1), device=input_ids.device).unsqueeze(0).repeat(input_ids.size(0), 1)
            )
            teacher_logits = self.teacher(teacher_input, src_key_padding_mask=attention_mask == 0)
            teacher_logits = self.student.output_proj(teacher_logits)
        
        loss = criterion(
            student_logits,
            teacher_logits,
            labels
        )
        
        loss.backward()
        self.optimizer.step()
        
        with torch.no_grad():
            student_probs = F.softmax(student_logits, dim=-1)
            teacher_probs = F.softmax(teacher_logits, dim=-1)
            
            student_preds = student_probs.argmax(dim=-1)
            accuracy = (student_preds == labels).float().mean()
            
            kl_div = F.kl_div(
                student_probs.log(),
                teacher_probs,
                reduction="batchmean"
            )
        
        return {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "kl_div": kl_div.item(),
            "temperature": temperature,
            "alpha": alpha
        }
    
    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        step: int
    ) -> Dict[str, float]:
        """
        Оцінка моделі.

        Args:
            dataloader: Завантажувач даних
            step: Номер кроку

        Returns:
            Словник з метриками
        """
        temperature, alpha = self.get_step_params(step)
        criterion = DistillationLoss(
            temperature=temperature,
            alpha=alpha
        )
        
        self.student.eval()
        total_loss = 0
        total_accuracy = 0
        total_kl_div = 0
        total_tokens = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["target_ids"].to(self.device)
                
                student_logits = self.student(input_ids, attention_mask)
                
                teacher_input = self.student.embed_tokens(input_ids) + self.student.pos_embed(
                    torch.arange(0, input_ids.size(1), device=input_ids.device).unsqueeze(0).repeat(input_ids.size(0), 1)
                )
                teacher_logits = self.teacher(teacher_input, src_key_padding_mask=attention_mask == 0)
                teacher_logits = self.student.output_proj(teacher_logits)
                
                loss = criterion(
                    student_logits,
                    teacher_logits,
                    labels
                )
                
                student_probs = F.softmax(student_logits, dim=-1)
                teacher_probs = F.softmax(teacher_logits, dim=-1)
                
                student_preds = student_probs.argmax(dim=-1)
                accuracy = (student_preds == labels).float().mean()
                
                kl_div = F.kl_div(
                    student_probs.log(),
                    teacher_probs,
                    reduction="batchmean"
                )
                
                total_loss += loss.item()
                total_accuracy += accuracy.item()
                total_kl_div += kl_div.item()
                total_tokens += attention_mask.sum().item()
                num_batches += 1
        
        return {
            "eval_loss": total_loss / num_batches,
            "eval_accuracy": total_accuracy / num_batches,
            "eval_kl_div": total_kl_div / num_batches,
            "eval_perplexity": torch.exp(torch.tensor(total_loss / total_tokens)).item(),
            "temperature": temperature,
            "alpha": alpha
        }
    
    def save_checkpoint(
        self,
        path: str,
        step: int,
        metrics: Dict[str, float]
    ):
        """
        Збереження чекпоінту.

        Args:
            path: Шлях для збереження
            step: Номер кроку
            metrics: Метрики
        """
        torch.save({
            "student_state": self.student.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "step": step,
            "metrics": metrics
        }, path)
    
    def load_checkpoint(
        self,
        path: str
    ) -> int:
        """
        Завантаження чекпоінту.

        Args:
            path: Шлях для завантаження

        Returns:
            Номер кроку
        """
        checkpoint = torch.load(path)
        self.student.load_state_dict(checkpoint["student_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        return checkpoint["step"] 