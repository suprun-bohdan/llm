"""
Скрипт для навчання моделі.
"""
import os
import argparse
import yaml
import torch
from pathlib import Path
from model.transformer import TransformerModel
from tokenizer.simple_tokenizer import SimpleTokenizer
from trainer.trainer import Trainer
from data.dataset import (
    create_dataloader,
    load_jsonl_dataset,
    split_dataset
)
from utils.logger import setup_logger


def parse_args():
    """Парсинг аргументів командного рядка."""
    parser = argparse.ArgumentParser(
        description="Навчання трансформера"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Шлях до конфігураційного файлу"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Шлях до файлу з даними (JSONL)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Директорія для збереження результатів"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        help="Шлях до чекпоінту для відновлення навчання"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Зерно для відтворення результатів"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """
    Завантаження конфігурації.

    Args:
        config_path: Шлях до конфігураційного файлу

    Returns:
        Словник з конфігурацією
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def setup_directories(output_dir: str) -> tuple:
    """
    Створення директорій для збереження результатів.

    Args:
        output_dir: Базова директорія

    Returns:
        Кортеж з шляхами до директорій
    """
    # Створення директорій
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    log_dir = os.path.join(output_dir, "logs")
    model_dir = os.path.join(output_dir, "model")
    
    for directory in [checkpoint_dir, log_dir, model_dir]:
        os.makedirs(directory, exist_ok=True)
    
    return checkpoint_dir, log_dir, model_dir


def main():
    """Головна функція."""
    # Парсинг аргументів
    args = parse_args()
    
    # Налаштування логування
    logger = setup_logger(
        name="train",
        log_dir=os.path.join(args.output_dir, "logs"),
        log_file="train.log"
    )
    
    # Завантаження конфігурації
    config = load_config(args.config)
    logger.info(f"Завантажено конфігурацію з {args.config}")
    
    # Створення директорій
    checkpoint_dir, log_dir, model_dir = setup_directories(args.output_dir)
    logger.info(f"Створено директорії в {args.output_dir}")
    
    # Встановлення зерна
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    logger.info(f"Встановлено зерно: {args.seed}")
    
    # Завантаження даних
    texts = load_jsonl_dataset(
        file_path=args.data,
        text_field=config["data"]["text_field"],
        max_samples=config["data"].get("max_samples")
    )
    logger.info(f"Завантажено {len(texts)} текстів")
    
    # Розділення на тренувальний та валідаційний набори
    train_texts, val_texts, _ = split_dataset(
        texts=texts,
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
        test_ratio=config["data"]["test_ratio"],
        seed=args.seed
    )
    logger.info(
        f"Розділено набори: "
        f"тренувальний ({len(train_texts)}), "
        f"валідаційний ({len(val_texts)})"
    )
    
    # Створення токенізатора
    tokenizer = SimpleTokenizer(
        vocab_size=config["tokenizer"]["vocab_size"],
        min_freq=config["tokenizer"]["min_freq"],
        special_tokens=config["tokenizer"]["special_tokens"]
    )
    
    # Навчання токенізатора
    tokenizer.train(train_texts)
    logger.info(
        f"Навчено токенізатор з розміром словника "
        f"{len(tokenizer.token_to_id)}"
    )
    
    # Створення завантажувачів
    train_dataloader = create_dataloader(
        texts=train_texts,
        tokenizer=tokenizer,
        max_seq_len=config["model"]["max_seq_len"],
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pad_token_id=tokenizer.token_to_id["<pad>"],
        bos_token_id=tokenizer.token_to_id["<bos>"],
        eos_token_id=tokenizer.token_to_id["<eos>"]
    )
    
    val_dataloader = create_dataloader(
        texts=val_texts,
        tokenizer=tokenizer,
        max_seq_len=config["model"]["max_seq_len"],
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pad_token_id=tokenizer.token_to_id["<pad>"],
        bos_token_id=tokenizer.token_to_id["<bos>"],
        eos_token_id=tokenizer.token_to_id["<eos>"]
    )
    logger.info("Створено завантажувачі даних")
    
    # Створення моделі
    model = TransformerModel(
        vocab_size=len(tokenizer.token_to_id),
        d_model=config["model"]["d_model"],
        n_heads=config["model"]["n_heads"],
        n_layers=config["model"]["n_layers"],
        d_ff=config["model"]["d_ff"],
        max_seq_len=config["model"]["max_seq_len"],
        dropout=config["model"]["dropout"]
    )
    logger.info("Створено модель")
    
    # Створення тренера
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=config["training"]["device"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        warmup_steps=config["training"]["warmup_steps"],
        max_grad_norm=config["training"]["max_grad_norm"],
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        save_every=config["training"]["save_every"],
        eval_every=config["training"]["eval_every"],
        max_epochs=config["training"]["max_epochs"],
        early_stopping_patience=config["training"]["early_stopping_patience"]
    )
    logger.info("Створено тренера")
    
    # Відновлення навчання
    if args.resume:
        trainer.load_checkpoint(args.resume)
        logger.info(f"Відновлено навчання з {args.resume}")
    
    # Навчання
    history = trainer.train()
    logger.info("Навчання завершено")
    
    # Збереження фінальної моделі
    model_path = os.path.join(model_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Збережено модель в {model_path}")
    
    # Збереження токенізатора
    tokenizer_path = os.path.join(model_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    logger.info(f"Збережено токенізатор в {tokenizer_path}")
    
    # Збереження конфігурації
    config_path = os.path.join(model_dir, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True)
    logger.info(f"Збережено конфігурацію в {config_path}")


if __name__ == "__main__":
    main() 