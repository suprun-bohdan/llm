"""
Script for model training.
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
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Transformer training"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data file (JSONL)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory for saving results"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint for resuming training"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducibility"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """
    Load configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def setup_directories(output_dir: str) -> tuple:
    """
    Create directories for saving results.

    Args:
        output_dir: Base directory

    Returns:
        Tuple with directory paths
    """
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    log_dir = os.path.join(output_dir, "logs")
    model_dir = os.path.join(output_dir, "model")
    
    for directory in [checkpoint_dir, log_dir, model_dir]:
        os.makedirs(directory, exist_ok=True)
    
    return checkpoint_dir, log_dir, model_dir


def main():
    """Main function."""
    args = parse_args()
    
    logger = setup_logger(
        name="train",
        log_dir=os.path.join(args.output_dir, "logs"),
        log_file="train.log"
    )
    
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    checkpoint_dir, log_dir, model_dir = setup_directories(args.output_dir)
    logger.info(f"Created directories in {args.output_dir}")
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    logger.info(f"Set seed: {args.seed}")
    
    texts = load_jsonl_dataset(
        file_path=args.data,
        text_field=config["data"]["text_field"],
        max_samples=config["data"].get("max_samples")
    )
    logger.info(f"Loaded {len(texts)} texts")
    
    train_texts, val_texts, _ = split_dataset(
        texts=texts,
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
        test_ratio=config["data"]["test_ratio"],
        seed=args.seed
    )
    logger.info(
        f"Split datasets: "
        f"training ({len(train_texts)}), "
        f"validation ({len(val_texts)})"
    )
    
    tokenizer = SimpleTokenizer(
        vocab_size=config["tokenizer"]["vocab_size"],
        min_freq=config["tokenizer"]["min_freq"],
        special_tokens=config["tokenizer"]["special_tokens"]
    )
    
    tokenizer.train(train_texts)
    logger.info(
        f"Trained tokenizer with vocabulary size "
        f"{len(tokenizer.token_to_id)}"
    )
    
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
    logger.info("Created data loaders")
    
    model = TransformerModel(
        vocab_size=len(tokenizer.token_to_id),
        d_model=config["model"]["d_model"],
        n_heads=config["model"]["n_heads"],
        n_layers=config["model"]["n_layers"],
        d_ff=config["model"]["d_ff"],
        max_seq_len=config["model"]["max_seq_len"],
        dropout=config["model"]["dropout"]
    )
    logger.info("Created model")
    
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
    logger.info("Created trainer")
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed training from {args.resume}")
    
    history = trainer.train()
    logger.info("Training completed")
    
    model_path = os.path.join(model_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved model to {model_path}")
    
    tokenizer_path = os.path.join(model_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    logger.info(f"Saved tokenizer to {tokenizer_path}")
    
    config_path = os.path.join(model_dir, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True)
    logger.info(f"Saved configuration to {config_path}")


if __name__ == "__main__":
    main() 