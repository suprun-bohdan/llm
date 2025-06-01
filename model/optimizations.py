"""
Оптимізації для моделі.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import math


class ReversibleBlock(nn.Module):
    """Оборотний блок для зменшення використання пам'яті."""
    
    def __init__(self, f: nn.Module, g: nn.Module):
        """
        Ініціалізація оборотного блоку.

        Args:
            f: Перша функція (наприклад, attention)
            g: Друга функція (наприклад, feed-forward)
        """
        super().__init__()
        self.f = f
        self.g = g
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Прямий прохід.

        Args:
            x1: Перший вхідний тензор
            x2: Другий вхідний тензор

        Returns:
            Кортеж з двох тензорів
        """
        y1 = x1 + self.f(x2)
        y2 = x2 + self.g(y1)
        return y1, y2
    
    def backward(self, y1: torch.Tensor, y2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Зворотній прохід.

        Args:
            y1: Перший вихідний тензор
            y2: Другий вихідний тензор

        Returns:
            Кортеж з двох тензорів
        """
        x2 = y2 - self.g(y1)
        x1 = y1 - self.f(x2)
        return x1, x2


class LowRankLinear(nn.Module):
    """Low-rank лінійний шар для зменшення кількості параметрів."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True
    ):
        """
        Ініціалізація low-rank шару.

        Args:
            in_features: Розмірність входу
            out_features: Розмірність виходу
            rank: Ранг розкладу
            bias: Використання зміщення
        """
        super().__init__()
        self.rank = rank
        self.u = nn.Linear(in_features, rank, bias=False)
        self.v = nn.Linear(rank, out_features, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямий прохід.

        Args:
            x: Вхідний тензор

        Returns:
            Вихідний тензор
        """
        return self.v(self.u(x))


class LoRALayer(nn.Module):
    """LoRA адаптер для fine-tuning."""
    
    def __init__(
        self,
        original: nn.Linear,
        rank: int = 4,
        scaling: float = 0.1,
        dropout: float = 0.1
    ):
        """
        Ініціалізація LoRA шару.

        Args:
            original: Оригінальний лінійний шар
            rank: Ранг LoRA
            scaling: Масштабування
            dropout: Dropout
        """
        super().__init__()
        self.original = original
        self.rank = rank
        self.scaling = scaling
        
        # LoRA адаптери
        self.lora_a = nn.Linear(
            original.in_features,
            rank,
            bias=False
        )
        self.lora_b = nn.Linear(
            rank,
            original.out_features,
            bias=False
        )
        self.dropout = nn.Dropout(dropout)
        
        # Ініціалізація
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямий прохід.

        Args:
            x: Вхідний тензор

        Returns:
            Вихідний тензор
        """
        return (
            self.original(x) +
            self.dropout(self.lora_b(self.lora_a(x))) * self.scaling
        )


class PerformerAttention(nn.Module):
    """Performer attention для ефективного обчислення."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        nb_features: int = 256,
        dropout: float = 0.1
    ):
        """
        Ініціалізація Performer attention.

        Args:
            d_model: Розмірність моделі
            n_heads: Кількість голів
            nb_features: Кількість випадкових особливостей
            dropout: Dropout
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.nb_features = nb_features
        self.head_dim = d_model // n_heads
        
        # Проекції
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Випадкові проекції для Performer
        self.register_buffer(
            "random_proj",
            torch.randn(nb_features, self.head_dim)
        )
    
    def _get_attention_scores(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Обчислення матриці уваги.

        Args:
            q: Запити
            k: Ключі
            mask: Маска

        Returns:
            Матриця уваги
        """
        # Проекція на випадкові особливості
        q_proj = F.linear(q, self.random_proj)
        k_proj = F.linear(k, self.random_proj)
        
        # Обчислення уваги
        scores = torch.matmul(q_proj, k_proj.transpose(-2, -1))
        scores = scores / math.sqrt(self.nb_features)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        
        return F.softmax(scores, dim=-1)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Прямий прохід.

        Args:
            x: Вхідний тензор
            mask: Маска

        Returns:
            Вихідний тензор
        """
        batch_size, seq_len, _ = x.size()
        
        # Проекції
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Транспонування для обчислення уваги
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Обчислення уваги
        scores = self._get_attention_scores(q, k, mask)
        scores = self.dropout(scores)
        
        # Згортка зі значеннями
        out = torch.matmul(scores, v)
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.d_model)
        
        return self.out_proj(out)


class MagnitudePruner:
    """Прунер на основі величини ваг."""
    
    def __init__(
        self,
        model: nn.Module,
        amount: float = 0.2,
        schedule: str = "gradual",
        start_epoch: int = 0,
        end_epoch: int = 10
    ):
        """
        Ініціалізація прунера.

        Args:
            model: Модель для прунінгу
            amount: Кількість ваг для прунінгу
            schedule: Графік прунінгу
            start_epoch: Початок прунінгу
            end_epoch: Кінець прунінгу
        """
        self.model = model
        self.amount = amount
        self.schedule = schedule
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        
        # Ініціалізація масок
        self.masks = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                self.masks[name] = torch.ones_like(module.weight)
    
    def _get_current_amount(self, epoch: int) -> float:
        """
        Отримання поточної кількості для прунінгу.

        Args:
            epoch: Поточна епоха

        Returns:
            Кількість для прунінгу
        """
        if epoch < self.start_epoch:
            return 0.0
        if epoch >= self.end_epoch:
            return self.amount
        
        if self.schedule == "gradual":
            return self.amount * (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
        return self.amount
    
    def step(self, epoch: int):
        """
        Крок прунінгу.

        Args:
            epoch: Поточна епоха
        """
        current_amount = self._get_current_amount(epoch)
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Обчислення порогу
                weights = module.weight.data.abs()
                threshold = torch.quantile(
                    weights.view(-1),
                    current_amount
                )
                
                # Оновлення маски
                self.masks[name] = (weights > threshold).float()
                
                # Застосування маски
                module.weight.data *= self.masks[name]


class MemoryBank:
    """Банк пам'яті для зберігання фактів."""
    
    def __init__(
        self,
        dim: int,
        max_size: int = 10000,
        similarity: str = "cosine"
    ):
        """
        Ініціалізація банку пам'яті.

        Args:
            dim: Розмірність векторів
            max_size: Максимальний розмір
            similarity: Метрика подібності
        """
        self.dim = dim
        self.max_size = max_size
        self.similarity = similarity
        
        self.vectors = []
        self.metadata = []
    
    def add(self, vector: torch.Tensor, metadata: dict):
        """
        Додавання вектора.

        Args:
            vector: Вектор
            metadata: Метадані
        """
        if len(self.vectors) >= self.max_size:
            self.vectors.pop(0)
            self.metadata.pop(0)
        
        self.vectors.append(vector)
        self.metadata.append(metadata)
    
    def retrieve(
        self,
        query: torch.Tensor,
        top_k: int = 5
    ) -> List[Tuple[torch.Tensor, dict, float]]:
        """
        Отримання найближчих векторів.

        Args:
            query: Запит
            top_k: Кількість результатів

        Returns:
            Список кортежів (вектор, метадані, схожість)
        """
        if not self.vectors:
            return []
        
        # Обчислення схожості
        vectors = torch.stack(self.vectors)
        if self.similarity == "cosine":
            similarities = F.cosine_similarity(
                query.unsqueeze(0),
                vectors,
                dim=1
            )
        else:  # euclidean
            similarities = -torch.norm(
                query.unsqueeze(0) - vectors,
                dim=1
            )
        
        # Отримання найкращих результатів
        top_indices = similarities.argsort(descending=True)[:top_k]
        
        return [
            (
                self.vectors[i],
                self.metadata[i],
                similarities[i].item()
            )
            for i in top_indices
        ]
    
    def save(self, path: str):
        """
        Збереження банку пам'яті.

        Args:
            path: Шлях для збереження
        """
        torch.save({
            "vectors": self.vectors,
            "metadata": self.metadata
        }, path)
    
    def load(self, path: str):
        """
        Завантаження банку пам'яті.

        Args:
            path: Шлях для завантаження
        """
        data = torch.load(path)
        self.vectors = data["vectors"]
        self.metadata = data["metadata"] 