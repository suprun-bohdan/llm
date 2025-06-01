"""
Дистиляція моделі.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math


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
        # Втрати дистиляції
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
        
        # Втрати навчання
        student_loss = self.ce(student_logits, labels)
        
        # Комбіновані втрати
        return (
            self.alpha * distillation_loss +
            (1 - self.alpha) * student_loss
        )


class DistillationTrainer:
    """Тренер для дистиляції."""
    
    def __init__(
        self,
        student: nn.Module,
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
        
        # Перенесення моделей на пристрій
        self.student.to(device)
        self.teacher.to(device)
        
        # Встановлення режиму вчителя
        self.teacher.eval()
    
    def train_step(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Крок навчання.

        Args:
            inputs: Входи
            labels: Мітки

        Returns:
            Словник з метриками
        """
        # Перенесення даних на пристрій
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        
        # Прямий прохід вчителя
        with torch.no_grad():
            teacher_logits = self.teacher(inputs)
        
        # Прямий прохід студента
        student_logits = self.student(inputs)
        
        # Обчислення втрат
        loss = self.criterion(
            student_logits,
            teacher_logits,
            labels
        )
        
        # Оптимізація
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Обчислення метрик
        with torch.no_grad():
            student_probs = F.softmax(student_logits, dim=-1)
            teacher_probs = F.softmax(teacher_logits, dim=-1)
            
            # Точність
            student_preds = student_probs.argmax(dim=-1)
            accuracy = (student_preds == labels).float().mean()
            
            # KL розбіжність
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
        num_batches = 0
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                # Перенесення даних на пристрій
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Прямі проходи
                teacher_logits = self.teacher(inputs)
                student_logits = self.student(inputs)
                
                # Обчислення втрат
                loss = self.criterion(
                    student_logits,
                    teacher_logits,
                    labels
                )
                
                # Обчислення метрик
                student_probs = F.softmax(student_logits, dim=-1)
                teacher_probs = F.softmax(teacher_logits, dim=-1)
                
                student_preds = student_probs.argmax(dim=-1)
                accuracy = (student_preds == labels).float().mean()
                
                kl_div = F.kl_div(
                    student_probs.log(),
                    teacher_probs,
                    reduction="batchmean"
                )
                
                # Оновлення статистики
                total_loss += loss.item()
                total_accuracy += accuracy.item()
                total_kl_div += kl_div.item()
                num_batches += 1
        
        # Обчислення середніх значень
        return {
            "loss": total_loss / num_batches,
            "accuracy": total_accuracy / num_batches,
            "kl_div": total_kl_div / num_batches
        }
    
    def save_checkpoint(self, path: str):
        """
        Збереження чекпоінту.

        Args:
            path: Шлях для збереження
        """
        torch.save({
            "student_state_dict": self.student.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }, path)
    
    def load_checkpoint(self, path: str):
        """
        Завантаження чекпоінту.

        Args:
            path: Шлях для завантаження
        """
        checkpoint = torch.load(path)
        self.student.load_state_dict(checkpoint["student_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


class ProgressiveDistillation:
    """Прогресивна дистиляція."""
    
    def __init__(
        self,
        student: nn.Module,
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
        self.device = device
        
        # Параметри для кожного кроку
        self.temperatures = torch.linspace(
            initial_temperature,
            final_temperature,
            num_steps
        )
        self.alphas = torch.linspace(
            initial_alpha,
            final_alpha,
            num_steps
        )
        
        # Перенесення моделей на пристрій
        self.student.to(device)
        self.teacher.to(device)
        
        # Встановлення режиму вчителя
        self.teacher.eval()
    
    def get_step_params(self, step: int) -> Tuple[float, float]:
        """
        Отримання параметрів для кроку.

        Args:
            step: Номер кроку

        Returns:
            Кортеж (температура, вага)
        """
        return (
            self.temperatures[step].item(),
            self.alphas[step].item()
        )
    
    def train_step(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        step: int
    ) -> Dict[str, float]:
        """
        Крок навчання.

        Args:
            inputs: Входи
            labels: Мітки
            step: Номер кроку

        Returns:
            Словник з метриками
        """
        # Отримання параметрів
        temperature, alpha = self.get_step_params(step)
        
        # Створення функції втрат
        criterion = DistillationLoss(
            temperature=temperature,
            alpha=alpha
        )
        
        # Перенесення даних на пристрій
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        
        # Прямий прохід вчителя
        with torch.no_grad():
            teacher_logits = self.teacher(inputs)
        
        # Прямий прохід студента
        student_logits = self.student(inputs)
        
        # Обчислення втрат
        loss = criterion(
            student_logits,
            teacher_logits,
            labels
        )
        
        # Оптимізація
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Обчислення метрик
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
    
    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        step: int
    ) -> Dict[str, float]:
        """
        Навчання епохи.

        Args:
            dataloader: Завантажувач даних
            step: Номер кроку

        Returns:
            Словник з метриками
        """
        self.student.train()
        total_loss = 0
        total_accuracy = 0
        total_kl_div = 0
        num_batches = 0
        
        for inputs, labels in dataloader:
            metrics = self.train_step(inputs, labels, step)
            
            total_loss += metrics["loss"]
            total_accuracy += metrics["accuracy"]
            total_kl_div += metrics["kl_div"]
            num_batches += 1
        
        return {
            "loss": total_loss / num_batches,
            "accuracy": total_accuracy / num_batches,
            "kl_div": total_kl_div / num_batches,
            "temperature": metrics["temperature"],
            "alpha": metrics["alpha"]
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
        self.student.eval()
        temperature, alpha = self.get_step_params(step)
        criterion = DistillationLoss(
            temperature=temperature,
            alpha=alpha
        )
        
        total_loss = 0
        total_accuracy = 0
        total_kl_div = 0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                # Перенесення даних на пристрій
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Прямі проходи
                teacher_logits = self.teacher(inputs)
                student_logits = self.student(inputs)
                
                # Обчислення втрат
                loss = criterion(
                    student_logits,
                    teacher_logits,
                    labels
                )
                
                # Обчислення метрик
                student_probs = F.softmax(student_logits, dim=-1)
                teacher_probs = F.softmax(teacher_logits, dim=-1)
                
                student_preds = student_probs.argmax(dim=-1)
                accuracy = (student_preds == labels).float().mean()
                
                kl_div = F.kl_div(
                    student_probs.log(),
                    teacher_probs,
                    reduction="batchmean"
                )
                
                # Оновлення статистики
                total_loss += loss.item()
                total_accuracy += accuracy.item()
                total_kl_div += kl_div.item()
                num_batches += 1
        
        return {
            "loss": total_loss / num_batches,
            "accuracy": total_accuracy / num_batches,
            "kl_div": total_kl_div / num_batches,
            "temperature": temperature,
            "alpha": alpha
        }
    
    def save_checkpoint(self, path: str, step: int):
        """
        Збереження чекпоінту.

        Args:
            path: Шлях для збереження
            step: Номер кроку
        """
        torch.save({
            "student_state_dict": self.student.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": step
        }, path)
    
    def load_checkpoint(self, path: str) -> int:
        """
        Завантаження чекпоінту.

        Args:
            path: Шлях для завантаження

        Returns:
            Номер кроку
        """
        checkpoint = torch.load(path)
        self.student.load_state_dict(checkpoint["student_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["step"] 