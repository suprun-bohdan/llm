"""
Головний скрипт для навчання моделі.
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
    """Датасет діалогів."""

    def __init__(
        self,
        data: list,
        tokenizer: SimpleTokenizer,
        max_seq_len: int
    ):
        """
        Ініціалізація.

        Args:
            data: Дані
            tokenizer: Токенізатор
            max_seq_len: Максимальна довжина послідовності
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        """Розмір датасету."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Отримання елемента.

        Args:
            idx: Індекс

        Returns:
            Словник з даними
        """
        item = self.data[idx]
        
        # Токенізація
        prompt_ids = self.tokenizer.encode(item["prompt"])
        response_ids = self.tokenizer.encode(item["response"])
        
        # Обмеження довжини
        prompt_ids = prompt_ids[:self.max_seq_len - 1]
        response_ids = response_ids[:self.max_seq_len - 1]
        
        # Додавання спеціальних токенів
        input_ids = (
            [self.tokenizer.token_to_id["<bos>"]] +
            prompt_ids +
            [self.tokenizer.token_to_id["<eos>"]] +
            response_ids +
            [self.tokenizer.token_to_id["<eos>"]]
        )
        
        # Створення міток (зсув на 1)
        labels = input_ids[1:] + [self.tokenizer.token_to_id["<pad>"]]
        
        # Паддінг
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
    Функція для згортки батчу.

    Args:
        batch: Батч

    Returns:
        Згортка батчу
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    
    return {
        "input_ids": input_ids,
        "labels": labels
    }


def main(args: argparse.Namespace) -> None:
    """
    Головна функція.

    Args:
        args: Аргументи командного рядка
    """
    # Завантаження конфігурації
    with open(args.config, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    config = Config.from_yaml(args.config)

    # Встановлення seed
    set_seed(args.seed)

    # Ініціалізація логера
    logger = Logger(
        log_dir=config.logging.log_dir,
        experiment_name=args.experiment_name
    )
    logger.log_config(config_dict)

    # Завантаження даних
    data = load_jsonl(args.data)
    train_data, val_data = split_data(
        data,
        val_split=config.data.val_split,
        seed=args.seed
    )

    # Ініціалізація токенізатора
    tokenizer = SimpleTokenizer(
        vocab_size=config.tokenizer.vocab_size,
        min_freq=config.tokenizer.min_freq,
        special_tokens=config.tokenizer.special_tokens
    )
    
    # Навчання токенізатора
    texts = [item["prompt"] + " " + item["response"] for item in train_data]
    tokenizer.train(texts)
    tokenizer.save("tokenizer/vocab.json")

    # Створення датасетів
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

    # Створення завантажувачів
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

    # Ініціалізація моделі
    model = TransformerModel(
        vocab_size=config.model.vocab_size,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        d_ff=config.model.d_ff,
        max_seq_len=config.model.max_seq_len,
        dropout=config.model.dropout
    )

    # Логування графу моделі
    logger.log_model_graph(model)

    # Ініціалізація тренера
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config_dict["training"],
        device=get_device()
    )

    # Завантаження чекпоінту, якщо вказано
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    # Навчання
    trainer.train(config.training.num_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Навчання мовної моделі")
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Шлях до конфігурації"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Шлях до даних"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Назва експерименту"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed для відтворюваності"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Кількість робочих процесів"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Шлях до чекпоінту"
    )
    
    args = parser.parse_args()
    main(args) 