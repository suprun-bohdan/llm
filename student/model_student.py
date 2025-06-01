"""
Student model implementation with LoRA adapters.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any
from model.transformer import TransformerModel
from model.optimizations import LoRALayer


class LoRALinear(nn.Module):
    def __init__(self, orig_linear: nn.Linear, rank: int = 4, lora_scaling: float = 0.1):
        super().__init__()
        self.orig = orig_linear
        self.rank = rank
        self.lora_a = nn.Linear(orig_linear.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, orig_linear.out_features, bias=False)
        self.scaling = lora_scaling
        
        # Делегуємо доступ до weight та bias оригінального лінійного шару
        self.weight = orig_linear.weight
        self.bias = orig_linear.bias

    def forward(self, x):
        return self.orig(x) + self.lora_b(self.lora_a(x)) * self.scaling


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, activation: str = "relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout)
        self.ffn = nn.Module()
        self.ffn.w1 = nn.Linear(d_model, d_ff)
        
        if activation == "gelu":
            self.ffn.act = lambda x: F.gelu(x)
        else:
            self.ffn.act = getattr(nn, activation)()
            
        self.ffn.w2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask=None):
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask)
        x = x + self.drop1(attn_out)
        x = self.norm1(x)

        ffn_out = self.ffn.w2(self.ffn.act(self.ffn.w1(x)))
        x = x + self.drop2(ffn_out)
        x = self.norm2(x)
        return x


class StudentModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        lora_rank: int,
        distill_alpha: float,
        use_lora: bool,
        ffn_rank: int,
        gradient_checkpointing: bool,
        mixed_precision: bool,
        max_seq_len: int = 256,
        activation: str = "relu"
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = ffn_rank
        self.dropout = dropout
        self.lora_rank = lora_rank
        self.distill_alpha = distill_alpha
        self.use_lora = use_lora
        self.gradient_checkpointing = gradient_checkpointing
        self.mixed_precision = mixed_precision
        self.max_seq_len = max_seq_len
        self.activation = activation

        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model, 
                n_heads=n_heads, 
                d_ff=self.d_ff, 
                dropout=dropout,
                activation=activation
            )
            for _ in range(n_layers)
        ])

        self.output_proj = nn.Linear(d_model, vocab_size)

        if self.use_lora:
            self._add_lora_adapters()

    def _add_lora_adapters(self):
        for block in self.layers:
            attn_out_proj = block.self_attn.out_proj
            lora_attn = LoRALinear(attn_out_proj, rank=self.lora_rank)
            block.self_attn.out_proj = lora_attn

            w1 = block.ffn.w1
            w2 = block.ffn.w2
            block.ffn.w1 = LoRALinear(w1, rank=self.lora_rank)
            block.ffn.w2 = LoRALinear(w2, rank=self.lora_rank)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.size()
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0).repeat(batch_size, 1)
        x = self.embed_tokens(input_ids) + self.pos_embed(positions)

        key_padding_mask = attention_mask == 0

        x = x.transpose(0, 1)
        for block in self.layers:
            x = block(x, src_key_padding_mask=key_padding_mask)
        x = x.transpose(0, 1)
        logits = self.output_proj(x)
        return logits

    def get_trainable_params(self) -> List[nn.Parameter]:
        return list(self.parameters())

    def freeze_base_model(self):
        for p in self.parameters():
            p.requires_grad = False 