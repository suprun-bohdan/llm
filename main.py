"""
Main script for model training.
"""
import os
import argparse
import yaml
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader, Dataset

from config import Config
from model.transformer import TransformerModel
from tokenizer.simple_tokenizer import SimpleTokenizer
from trainer.train_loop import Trainer
from utils.helpers import set_seed, load_jsonl, split_data, get_device
from utils.logger import Logger


class DialogueDataset(Dataset):
    """Dialogue dataset."""

    def __init__(
        self,
        data: list,
        tokenizer: SimpleTokenizer,
        max_seq_len: int
    ):
        """
        Initialize.

        Args:
            data: Data
            tokenizer: Tokenizer
            max_seq_len: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        """Dataset size."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item.

        Args:
            idx: Index

        Returns:
            Dictionary with data
        """
        item = self.data[idx]
        
        prompt_ids = self.tokenizer.encode(item["prompt"])
        response_ids = self.tokenizer.encode(item["response"])
        
        prompt_ids = prompt_ids[:self.max_seq_len - 1]
        response_ids = response_ids[:self.max_seq_len - 1]
        
        input_ids = (
            [self.tokenizer.token_to_id["<bos>"]] +
            prompt_ids +
            [self.tokenizer.token_to_id["<eos>"]] +
            response_ids +
            [self.tokenizer.token_to_id["<eos>"]]
        )
        
        labels = input_ids[1:] + [self.tokenizer.token_to_id["<pad>"]]
        
        if len(input_ids) < self.max_seq_len:
            pad_len = self.max_seq_len - len(input_ids)
            input_ids = input_ids + [self.tokenizer.token_to_id["<pad>"]] * pad_len
            labels = labels + [self.tokenizer.token_to_id["<pad>"]] * pad_len
        
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels)
        }


def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    """
    Batch collation function.

    Args:
        batch: Batch

    Returns:
        Collated batch
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    
    return {
        "input_ids": input_ids,
        "labels": labels
    }


def main(args: argparse.Namespace) -> None:
    """
    Main function.

    Args:
        args: Command line arguments
    """
    with open(args.config, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    config = Config.from_yaml(args.config)

    set_seed(args.seed)

    logger = Logger(
        log_dir=config.logging.log_dir,
        experiment_name=args.experiment_name
    )
    logger.log_config(config_dict)

    data = load_jsonl(args.data)
    train_data, val_data = split_data(
        data,
        val_split=config.data.val_split,
        seed=args.seed
    )

    tokenizer = SimpleTokenizer(
        vocab_size=config.tokenizer.vocab_size,
        min_freq=config.tokenizer.min_freq,
        special_tokens=config.tokenizer.special_tokens
    )
    
    texts = [item["prompt"] + " " + item["response"] for item in train_data]
    tokenizer.train(texts)
    tokenizer.save("tokenizer/vocab.json")

    train_dataset = DialogueDataset(
        train_data,
        tokenizer,
        config.model.max_seq_len
    )
    val_dataset = DialogueDataset(
        val_data,
        tokenizer,
        config.model.max_seq_len
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )

    model = TransformerModel(
        vocab_size=config.model.vocab_size,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        d_ff=config.model.d_ff,
        max_seq_len=config.model.max_seq_len,
        dropout=config.model.dropout
    )

    logger.log_model_graph(model)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config_dict["training"],
        device=get_device()
    )

    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    trainer.train(config.training.num_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Language model training")
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Experiment name"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducibility"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint"
    )
    
    args = parser.parse_args()
    main(args) 