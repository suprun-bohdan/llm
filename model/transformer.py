"""
Transformer implementation from scratch.
"""
import math
from typing import Optional, Tuple, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np


class PositionalEncoding(nn.Module):
    """Positional encoding."""

    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
        """
        Initialize.

        Args:
            d_model: Model dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]

        Returns:
            Tensor with positional encoding
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-head attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        """
        Initialize.

        Args:
            d_model: Model dimension
            n_heads: Number of heads
            dropout: Dropout
            bias: Use bias
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

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
            q: Queries [batch_size, seq_len, d_model]
            k: Keys [batch_size, seq_len, d_model]
            v: Values [batch_size, seq_len, d_model]
            mask: Mask [batch_size, seq_len, seq_len]

        Returns:
            Output [batch_size, seq_len, d_model]
        """
        batch_size = q.size(0)

        q = self.q_proj(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Two-layer perceptron."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        """
        Initialize.

        Args:
            d_model: Model dimension
            d_ff: Hidden layer dimension
            dropout: Dropout
            bias: Use bias
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=bias)
        self.linear2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input [batch_size, seq_len, d_model]

        Returns:
            Output [batch_size, seq_len, d_model]
        """
        x = F.gelu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        """
        Initialize.

        Args:
            d_model: Model dimension
            n_heads: Number of heads
            d_ff: FFN dimension
            dropout: Dropout
            bias: Use bias
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
            x: Input [batch_size, seq_len, d_model]
            mask: Mask [batch_size, seq_len, seq_len]

        Returns:
            Output [batch_size, seq_len, d_model]
        """
        attn_output = self.self_attn(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)

        return x


class TransformerModel(nn.Module):
    """Full transformer model."""

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
        Initialize.

        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            n_heads: Number of heads
            n_layers: Number of layers
            d_ff: FFN dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout
            bias: Use bias
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, bias)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=bias)
        
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights."""
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
            x: Input [batch_size, seq_len]
            mask: Mask [batch_size, seq_len, seq_len]

        Returns:
            Output [batch_size, seq_len, vocab_size]
        """
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.norm(x)
        x = self.head(x)
        
        return x

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create causal mask.

        Args:
            seq_len: Sequence length
            device: Device

        Returns:
            Mask [seq_len, seq_len]
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        mask = mask.masked_fill(mask == 0, float(0.0))
        return mask

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
        Generate text.

        Args:
            prompt: Input text
            tokenizer: Tokenizer
            max_len: Maximum length
            temperature: Sampling temperature
            top_k: Number of top tokens
            top_p: Nucleus sampling threshold
            beam_size: Beam size

        Returns:
            Generated text
        """
        self.eval()
        device = next(self.parameters()).device
        
        input_ids = torch.tensor(
            [tokenizer.encode(prompt)],
            device=device
        )
        
        with torch.no_grad():
            for _ in range(max_len):
                mask = self._create_causal_mask(
                    input_ids.size(1),
                    device
                )
                
                logits = self(input_ids, mask)[:, -1, :]
                logits = logits / temperature
                
                if top_k is not None:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float("-inf")
                
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float("-inf")
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                if next_token.item() == tokenizer.token_to_id["<eos>"]:
                    break
        
        return tokenizer.decode(input_ids[0].tolist())

    def save(self, path: str) -> None:
        """
        Save model.

        Args:
            path: Path to file
        """
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "TransformerModel":
        """
        Load model.

        Args:
            path: Path to file
            **kwargs: Model parameters

        Returns:
            Loaded model
        """
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path))
        return model 