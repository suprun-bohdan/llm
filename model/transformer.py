"""
Реалізація трансформера з нуля.
"""
import math
from typing import Optional, Tuple, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np


class PositionalEncoding(nn.Module):
    """Позиційне кодування."""

    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
        """
        Ініціалізація.

        Args:
            d_model: Розмірність моделі
            max_seq_len: Максимальна довжина послідовності
            dropout: Ймовірність dropout
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Створення матриці позиційного кодування
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_seq_len, d_model]
        
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Тензор форми [batch_size, seq_len, d_model]

        Returns:
            Тензор з позиційним кодуванням
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Багатоголова увага."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        """
        Ініціалізація.

        Args:
            d_model: Розмірність моделі
            n_heads: Кількість голів
            dropout: Dropout
            bias: Використовувати зміщення
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model має ділитися на n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Проекції
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(p=dropout)
        self.scale = 1.0 / math.sqrt(self.d_k)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            q: Запити [batch_size, seq_len, d_model]
            k: Ключі [batch_size, seq_len, d_model]
            v: Значення [batch_size, seq_len, d_model]
            mask: Маска [batch_size, seq_len, seq_len]

        Returns:
            Вихід [batch_size, seq_len, d_model]
        """
        batch_size = q.size(0)

        # Проекції
        q = self.q_proj(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Обчислення уваги
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Згортка зі значеннями
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Двошаровий перцептрон."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        """
        Ініціалізація.

        Args:
            d_model: Розмірність моделі
            d_ff: Розмірність прихованого шару
            dropout: Dropout
            bias: Використовувати зміщення
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=bias)
        self.linear2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Вхід [batch_size, seq_len, d_model]

        Returns:
            Вихід [batch_size, seq_len, d_model]
        """
        x = F.gelu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """Блок трансформера."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        """
        Ініціалізація.

        Args:
            d_model: Розмірність моделі
            n_heads: Кількість голів
            d_ff: Розмірність FFN
            dropout: Dropout
            bias: Використовувати зміщення
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout, bias)
        self.feed_forward = FeedForward(d_model, d_ff, dropout, bias)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Вхід [batch_size, seq_len, d_model]
            mask: Маска [batch_size, seq_len, seq_len]

        Returns:
            Вихід [batch_size, seq_len, d_model]
        """
        # Self-attention
        attn_output = self.self_attn(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Feed-forward
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)

        return x


class TransformerModel(nn.Module):
    """Повна модель трансформера."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        bias: bool = True
    ):
        """
        Ініціалізація.

        Args:
            vocab_size: Розмір словника
            d_model: Розмірність моделі
            n_heads: Кількість голів
            n_layers: Кількість шарів
            d_ff: Розмірність FFN
            max_seq_len: Максимальна довжина послідовності
            dropout: Dropout
            bias: Використовувати зміщення
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, bias)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=bias)
        
        # Ініціалізація ваг
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Ініціалізація ваг."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Вхід [batch_size, seq_len]
            mask: Маска [batch_size, seq_len, seq_len]

        Returns:
            Логіти [batch_size, seq_len, vocab_size]
        """
        # Ембедінги + позиційне кодування
        x = self.embedding(x)
        x = self.pos_encoding(x)

        # Трансформерні блоки
        for block in self.blocks:
            x = block(x, mask)

        # Вихідний шар
        x = self.norm(x)
        x = self.output(x)

        return x

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Створення каузальної маски.

        Args:
            seq_len: Довжина послідовності
            device: Пристрій

        Returns:
            Маска [seq_len, seq_len]
        """
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device),
            diagonal=1
        ).bool()
        return ~mask

    def generate(
        self,
        prompt: str,
        tokenizer: Any,
        max_len: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        beam_size: int = 1
    ) -> str:
        """
        Генерація тексту.

        Args:
            prompt: Початковий текст
            tokenizer: Токенізатор
            max_len: Максимальна довжина
            temperature: Температура
            top_k: Top-k sampling
            top_p: Top-p sampling
            beam_size: Розмір beam search

        Returns:
            Згенерований текст
        """
        # TODO: Реалізувати генерацію
        return ""

    def save(self, path: str) -> None:
        """
        Збереження моделі.

        Args:
            path: Шлях до файлу
        """
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "TransformerModel":
        """
        Завантаження моделі.

        Args:
            path: Шлях до файлу
            **kwargs: Параметри моделі

        Returns:
            Завантажена модель
        """
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path))
        return model 