"""
Helper functions.
"""
import os
import json
import random
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple


def set_seed(seed: int) -> None:
    """
    Set seed for reproducibility.

    Args:
        seed: Seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_mask(
    seq_len: int,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Create mask for self-attention.

    Args:
        seq_len: Sequence length
        device: Device

    Returns:
        Mask [seq_len, seq_len]
    """
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device),
        diagonal=1
    ).bool()
    return ~mask


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Load JSONL file.

    Args:
        path: Path to file

    Returns:
        List of dictionaries
    """
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], path: str) -> None:
    """
    Save to JSONL file.

    Args:
        data: List of dictionaries
        path: Path to file
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def split_data(
    data: List[Dict[str, Any]],
    val_split: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split data into training and validation sets.

    Args:
        data: Data
        val_split: Validation set ratio
        seed: Seed for reproducibility

    Returns:
        Training and validation sets
    """
    if seed is not None:
        random.seed(seed)

    random.shuffle(data)
    split_idx = int(len(data) * (1 - val_split))
    return data[:split_idx], data[split_idx:]


def compute_perplexity(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100
) -> float:
    """
    Compute perplexity.

    Args:
        logits: Logits [batch_size, seq_len, vocab_size]
        labels: Labels [batch_size, seq_len]
        ignore_index: Index to ignore

    Returns:
        Perplexity
    """
    loss = torch.nn.CrossEntropyLoss(
        ignore_index=ignore_index,
        reduction="none"
    )(
        logits.view(-1, logits.size(-1)),
        labels.view(-1)
    )
    return torch.exp(loss.mean()).item()


def get_device() -> torch.device:
    """
    Get device.

    Returns:
        Device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count model parameters.

    Args:
        model: Model

    Returns:
        Number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 