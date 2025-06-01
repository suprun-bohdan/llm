"""
Script for text generation.
"""
import os
import argparse
import yaml
import torch
from pathlib import Path
from model.transformer import TransformerModel
from tokenizer.simple_tokenizer import SimpleTokenizer
from utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Text generation"
    )
    
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Model directory"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Initial text"
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=100,
        help="Maximum text length"
    )
    
    parser.add_argument(
        "--strategy",
        type=str,
        default="greedy",
        choices=["greedy", "top_k", "top_p", "beam"],
        help="Sampling strategy"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature"
    )
    
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Number of top tokens"
    )
    
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Threshold for nucleus sampling"
    )
    
    parser.add_argument(
        "--beam_size",
        type=int,
        default=1,
        help="Beam size"
    )
    
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="Number of variants"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducibility"
    )
    
    return parser.parse_args()


def load_model_and_tokenizer(model_dir: str) -> tuple:
    """
    Load model and tokenizer.

    Args:
        model_dir: Model directory

    Returns:
        Tuple with model and tokenizer
    """
    config_path = os.path.join(model_dir, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    tokenizer_path = os.path.join(model_dir, "tokenizer.json")
    tokenizer = SimpleTokenizer(
        vocab_size=config["tokenizer"]["vocab_size"],
        min_freq=config["tokenizer"]["min_freq"],
        special_tokens=config["tokenizer"]["special_tokens"]
    )
    tokenizer.load(tokenizer_path)
    
    model = TransformerModel(
        vocab_size=len(tokenizer.token_to_id),
        d_model=config["model"]["d_model"],
        n_heads=config["model"]["n_heads"],
        n_layers=config["model"]["n_layers"],
        d_ff=config["model"]["d_ff"],
        max_seq_len=config["model"]["max_seq_len"],
        dropout=0.0
    )
    
    model_path = os.path.join(model_dir, "model.pt")
    model.load_state_dict(torch.load(model_path))
    
    return model, tokenizer


def main():
    """Main function."""
    args = parse_args()
    
    logger = setup_logger(
        name="generate",
        log_dir=os.path.join(args.model_dir, "logs"),
        log_file="generate.log"
    )
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    logger.info(f"Set seed: {args.seed}")
    
    model, tokenizer = load_model_and_tokenizer(args.model_dir)
    logger.info("Loaded model and tokenizer")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    logger.info(f"Model moved to {device}")
    
    with torch.no_grad():
        texts = model.generate(
            input_ids=torch.tensor(
                [tokenizer.encode(args.prompt)],
                device=device
            ),
            max_length=args.max_length,
            strategy=args.strategy,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            beam_size=args.beam_size,
            num_return_sequences=args.num_return_sequences,
            pad_token_id=tokenizer.token_to_id["<pad>"],
            eos_token_id=tokenizer.token_to_id["<eos>"]
        )
    
    texts = [tokenizer.decode(ids.tolist()) for ids in texts]
    
    logger.info("Generated texts:")
    for i, text in enumerate(texts, 1):
        logger.info(f"\nVariant {i}:\n{text}")


if __name__ == "__main__":
    main() 