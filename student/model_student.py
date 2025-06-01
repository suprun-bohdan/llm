"""
Student model implementation with LoRA adapters.
"""
import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any
from model.transformer import TransformerModel
from model.optimizations import LoRALayer


class StudentModel(TransformerModel):
    """Student model with LoRA adapters."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.1,
        use_lora: bool = True,
        lora_rank: int = 4,
        lora_scaling: float = 0.1,
        shared_weights: bool = False,
        low_rank_ffn: bool = False,
        ffn_rank: Optional[int] = None
    ):
        """
        Initialize student model.

        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            use_lora: Whether to use LoRA adapters
            lora_rank: LoRA rank
            lora_scaling: LoRA scaling factor
            shared_weights: Whether to use weight sharing
            low_rank_ffn: Whether to use low-rank FFN
            ffn_rank: FFN rank for low-rank decomposition
        """
        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_scaling = lora_scaling
        self.shared_weights = shared_weights
        self.low_rank_ffn = low_rank_ffn
        self.ffn_rank = ffn_rank or d_ff // 4
        
        if shared_weights:
            self.shared_attention = nn.Linear(d_model, d_model)
            self.shared_ffn = nn.Linear(d_model, d_ff)
            self.layer_scales = nn.Parameter(torch.ones(n_layers, 2))
        
        if low_rank_ffn:
            self.ffn_u = nn.Parameter(torch.randn(n_layers, d_ff, self.ffn_rank))
            self.ffn_v = nn.Parameter(torch.randn(n_layers, self.ffn_rank, d_model))
        
        if use_lora:
            self._add_lora_adapters()

    def _add_lora_adapters(self):
        """Add LoRA adapters to linear layers."""
        for i, block in enumerate(self.blocks):
            # Add LoRA to attention layers
            block.attention.q_proj = LoRALayer(
                block.attention.q_proj,
                rank=self.lora_rank,
                scaling=self.lora_scaling
            )
            block.attention.k_proj = LoRALayer(
                block.attention.k_proj,
                rank=self.lora_rank,
                scaling=self.lora_scaling
            )
            block.attention.v_proj = LoRALayer(
                block.attention.v_proj,
                rank=self.lora_rank,
                scaling=self.lora_scaling
            )
            block.attention.o_proj = LoRALayer(
                block.attention.o_proj,
                rank=self.lora_rank,
                scaling=self.lora_scaling
            )
            
            # Add LoRA to FFN layers
            block.ffn.w1 = LoRALayer(
                block.ffn.w1,
                rank=self.lora_rank,
                scaling=self.lora_scaling
            )
            block.ffn.w2 = LoRALayer(
                block.ffn.w2,
                rank=self.lora_rank,
                scaling=self.lora_scaling
            )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor
            attention_mask: Attention mask

        Returns:
            Output tensor
        """
        x = self.embedding(x)
        x = self.pos_encoding(x)
        
        for i, block in enumerate(self.blocks):
            if self.shared_weights:
                # Apply shared weights with layer-specific scaling
                attn_scale = self.layer_scales[i, 0]
                ffn_scale = self.layer_scales[i, 1]
                
                # Attention
                q = attn_scale * self.shared_attention(x)
                k = attn_scale * self.shared_attention(x)
                v = attn_scale * self.shared_attention(x)
                
                # FFN
                if self.low_rank_ffn:
                    ffn_out = ffn_scale * (x @ self.ffn_u[i] @ self.ffn_v[i])
                else:
                    ffn_out = ffn_scale * self.shared_ffn(x)
            else:
                # Use original block
                x = block(x, attention_mask)
                continue
            
            # Apply attention and FFN
            attn_out = block.attention(q, k, v, attention_mask)
            x = x + self.dropout(attn_out)
            x = x + self.dropout(ffn_out)
            x = block.norm2(x)
        
        x = self.output(x)
        return x

    def get_trainable_params(self) -> List[nn.Parameter]:
        """
        Get trainable parameters.

        Returns:
            List of trainable parameters
        """
        if self.use_lora:
            return [p for p in self.parameters() if p.requires_grad]
        return list(self.parameters())

    def freeze_base_model(self):
        """Freeze base model parameters."""
        for p in self.parameters():
            p.requires_grad = False
        
        if self.use_lora:
            # Unfreeze LoRA parameters
            for module in self.modules():
                if isinstance(module, LoRALayer):
                    for p in module.parameters():
                        p.requires_grad = True 