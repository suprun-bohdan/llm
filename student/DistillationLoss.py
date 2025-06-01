import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, temperature: float = 2.0, alpha: float = 0.5):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Обчислення крос-ентропійних втрат
        ce_loss = self.ce_loss(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))
        
        # Обчислення KL-дивергенції
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        kl_loss = self.kl_loss(student_log_probs, teacher_probs) * (self.temperature ** 2)
        
        # Загальні втрати
        total_loss = self.alpha * ce_loss + (1 - self.alpha) * kl_loss
        
        return total_loss