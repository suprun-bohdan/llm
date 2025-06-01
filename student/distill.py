"""
Knowledge distillation implementation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
from student.model_student import StudentModel


class DistillationTrainer:
    """Knowledge distillation trainer."""

    def __init__(
        self,
        student: StudentModel,
        teacher: nn.Module,
        alpha: float = 0.5,
        temperature: float = 2.0,
        device: str = "cuda"
    ):
        """
        Initialize distillation trainer.

        Args:
            student: Student model
            teacher: Teacher model
            alpha: Weight for distillation loss
            temperature: Temperature for softmax
            device: Device to use
        """
        self.student = student
        self.teacher = teacher
        self.alpha = alpha
        self.temperature = temperature
        self.device = device
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        
        self.student.to(device)
        self.teacher.to(device)
        self.teacher.eval()

    def compute_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute distillation loss.

        Args:
            student_logits: Student model logits
            teacher_logits: Teacher model logits
            labels: Target labels

        Returns:
            Total loss and loss components
        """
        # Compute cross-entropy loss
        ce_loss = self.ce_loss(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))
        
        # Compute KL divergence loss
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        kl_loss = self.kl_loss(student_log_probs, teacher_probs) * (self.temperature ** 2)
        
        # Combine losses
        total_loss = self.alpha * ce_loss + (1 - self.alpha) * kl_loss
        
        return total_loss, {
            "ce_loss": ce_loss.item(),
            "kl_loss": kl_loss.item(),
            "total_loss": total_loss.item()
        }

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Perform one training step.

        Args:
            batch: Input batch
            optimizer: Optimizer

        Returns:
            Loss components
        """
        self.student.train()
        optimizer.zero_grad()
        
        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["target_ids"].to(self.device)
        
        # Get student predictions
        student_logits = self.student(input_ids, attention_mask)
        
        # Get teacher predictions
        with torch.no_grad():
            teacher_logits = self.teacher(input_ids, attention_mask)
        
        # Compute loss
        loss, loss_components = self.compute_loss(student_logits, teacher_logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return loss_components

    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate student model.

        Args:
            dataloader: Evaluation dataloader

        Returns:
            Evaluation metrics
        """
        self.student.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["target_ids"].to(self.device)
                
                # Get predictions
                student_logits = self.student(input_ids, attention_mask)
                teacher_logits = self.teacher(input_ids, attention_mask)
                
                # Compute loss
                loss, _ = self.compute_loss(student_logits, teacher_logits, labels)
                
                # Update metrics
                total_loss += loss.item() * input_ids.size(0)
                total_tokens += attention_mask.sum().item()
        
        return {
            "eval_loss": total_loss / len(dataloader),
            "eval_perplexity": torch.exp(torch.tensor(total_loss / total_tokens)).item()
        }

    def save_checkpoint(
        self,
        path: str,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float]
    ):
        """
        Save training checkpoint.

        Args:
            path: Checkpoint path
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Training metrics
        """
        torch.save({
            "student_state": self.student.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics
        }, path)

    def load_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Tuple[int, Dict[str, float]]:
        """
        Load training checkpoint.

        Args:
            path: Checkpoint path
            optimizer: Optimizer to load state into

        Returns:
            Epoch and metrics
        """
        checkpoint = torch.load(path)
        self.student.load_state_dict(checkpoint["student_state"])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        
        return checkpoint["epoch"], checkpoint["metrics"] 