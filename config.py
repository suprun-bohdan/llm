"""
Module for working with model configuration.
"""
import os
import yaml
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class ModelConfig:
    """Model configuration."""
    vocab_size: int
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    max_seq_len: int
    dropout: float


@dataclass
class TokenizerConfig:
    """Tokenizer configuration."""
    vocab_size: int
    min_freq: int
    special_tokens: List[str]


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    gradient_clip_val: float
    gradient_accumulation_steps: int
    eval_steps: int
    save_steps: int
    logging_steps: int
    checkpoint_dir: str


@dataclass
class GenerationConfig:
    """Generation configuration."""
    max_len: int
    temperature: float
    top_k: int
    top_p: float
    beam_size: int


@dataclass
class DataConfig:
    """Data configuration."""
    val_split: float


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_dir: str


@dataclass
class Config:
    """General configuration."""
    tokenizer: TokenizerConfig
    model: ModelConfig
    training: TrainingConfig
    generation: GenerationConfig
    data: DataConfig
    logging: LoggingConfig

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """
        Load configuration from YAML.

        Args:
            path: Path to file

        Returns:
            Configuration
        """
        with open(path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        return cls(
            tokenizer=TokenizerConfig(**config_dict["tokenizer"]),
            model=ModelConfig(**config_dict["model"]),
            training=TrainingConfig(**config_dict["training"]),
            generation=GenerationConfig(**config_dict["generation"]),
            data=DataConfig(**config_dict["data"]),
            logging=LoggingConfig(**config_dict["logging"])
        )

    def to_yaml(self, path: str) -> None:
        """
        Save configuration to YAML.

        Args:
            path: Path to file
        """
        config_dict = {
            "tokenizer": self.tokenizer.__dict__,
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "generation": self.generation.__dict__,
            "data": self.data.__dict__,
            "logging": self.logging.__dict__
        }

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, allow_unicode=True) 