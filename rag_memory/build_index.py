"""
Script for building Faiss index.
"""
import torch
import argparse
import json
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm
from rag_memory.memory_bank import MemoryBank
from model.transformer import TransformerModel
from tokenizer.bpe_pq_tokenizer import BPETokenizer


def load_data(data_path: str) -> List[Dict]:
    """
    Load data from JSONL file.

    Args:
        data_path: Path to JSONL file

    Returns:
        List of data items
    """
    data = []
    with open(data_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def get_embeddings(
    model: TransformerModel,
    tokenizer: BPETokenizer,
    data: List[Dict],
    batch_size: int = 32,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Get embeddings for data.

    Args:
        model: Transformer model
        tokenizer: BPE tokenizer
        data: List of data items
        batch_size: Batch size
        device: Device to use

    Returns:
        Embedding matrix
    """
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(data), batch_size)):
            batch = data[i:i + batch_size]
            
            # Tokenize
            texts = [item["text"] for item in batch]
            encodings = [tokenizer.encode(text) for text in texts]
            
            # Stack
            input_ids = torch.stack([e["input_ids"] for e in encodings]).to(device)
            attention_mask = torch.stack([e["attention_mask"] for e in encodings]).to(device)
            
            # Get embeddings
            outputs = model(input_ids, attention_mask)
            batch_emb = outputs.mean(dim=1)  # Mean pooling
            
            embeddings.append(batch_emb.cpu())
    
    return torch.cat(embeddings, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--nlist", type=int, default=100)
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--nbits", type=int, default=8)
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    data = load_data(args.data_path)
    
    # Load model
    print("Loading model...")
    model = TransformerModel.from_pretrained(args.model_path)
    model.to(args.device)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = BPETokenizer()
    tokenizer.load(args.tokenizer_path)
    
    # Get embeddings
    print("Getting embeddings...")
    embeddings = get_embeddings(
        model,
        tokenizer,
        data,
        args.batch_size,
        args.device
    )
    
    # Build index
    print("Building index...")
    memory_bank = MemoryBank(
        dim=args.dim,
        nlist=args.nlist,
        m=args.m,
        nbits=args.nbits,
        device=args.device
    )
    
    # Extract metadata
    metadata = [{"text": item["text"], "id": i} for i, item in enumerate(data)]
    
    # Build index
    memory_bank.build(embeddings, metadata)
    
    # Save index
    print("Saving index...")
    memory_bank.save(
        output_dir / "index.faiss",
        output_dir / "metadata.json"
    )
    
    # Print stats
    stats = memory_bank.get_stats()
    print("\nIndex statistics:")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main() 