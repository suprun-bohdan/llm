"""
BPE tokenizer and PQ embedding implementation.
"""
import torch
import torch.nn as nn
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from collections import defaultdict


class BPETokenizer:
    """BPE tokenizer implementation."""

    def __init__(
        self,
        vocab_size: int = 4096,
        min_frequency: int = 2,
        special_tokens: List[str] = None
    ):
        """
        Initialize BPE tokenizer.

        Args:
            vocab_size: Vocabulary size
            min_frequency: Minimum token frequency
            special_tokens: List of special tokens
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = special_tokens or ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        
        self.tokenizer = Tokenizer(models.BPE())
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        
        # Add special tokens
        self.tokenizer.add_special_tokens(self.special_tokens)
        
        # Add post-processor
        self.tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B [SEP]",
            special_tokens=[
                ("[CLS]", self.tokenizer.token_to_id("[CLS]")),
                ("[SEP]", self.tokenizer.token_to_id("[SEP]"))
            ]
        )

    def train(
        self,
        files: List[str],
        min_frequency: Optional[int] = None
    ):
        """
        Train tokenizer.

        Args:
            files: List of training files
            min_frequency: Minimum token frequency
        """
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=min_frequency or self.min_frequency,
            special_tokens=self.special_tokens
        )
        
        self.tokenizer.train(files, trainer)

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text.

        Args:
            text: Input text
            add_special_tokens: Whether to add special tokens

        Returns:
            Dictionary with input_ids and attention_mask
        """
        encoding = self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens
        )
        
        return {
            "input_ids": torch.tensor(encoding.ids),
            "attention_mask": torch.tensor(encoding.attention_mask)
        }

    def decode(
        self,
        ids: Union[List[int], torch.Tensor]
    ) -> str:
        """
        Decode token ids.

        Args:
            ids: Token ids

        Returns:
            Decoded text
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        
        return self.tokenizer.decode(ids)

    def save(self, path: str):
        """
        Save tokenizer.

        Args:
            path: Save path
        """
        self.tokenizer.save(path)

    def load(self, path: str):
        """
        Load tokenizer.

        Args:
            path: Load path
        """
        self.tokenizer = Tokenizer.from_file(path)


class PQEmbedding(nn.Module):
    """Product Quantization embedding layer."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        m: int = 4,
        k: int = 256,
        padding_idx: Optional[int] = None
    ):
        """
        Initialize PQ embedding.

        Args:
            num_embeddings: Number of embeddings
            embedding_dim: Embedding dimension
            m: Number of subvectors
            k: Number of centroids per subvector
            padding_idx: Padding index
        """
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.m = m
        self.k = k
        self.padding_idx = padding_idx
        
        # Check if embedding_dim is divisible by m
        assert embedding_dim % m == 0, "embedding_dim must be divisible by m"
        self.sub_dim = embedding_dim // m
        
        # Initialize codebooks
        self.codebooks = nn.Parameter(
            torch.randn(m, k, self.sub_dim)
        )
        
        # Initialize codes
        self.codes = nn.Parameter(
            torch.randint(0, k, (num_embeddings, m))
        )
        
        # Initialize padding
        if padding_idx is not None:
            with torch.no_grad():
                self.codes[padding_idx].zero_()

    def _get_embeddings(self) -> torch.Tensor:
        """
        Get full embeddings from codes.

        Returns:
            Embedding matrix
        """
        # Reshape codes for indexing
        codes = self.codes.view(-1, self.m, 1, 1)
        
        # Get embeddings from codebooks
        embeddings = torch.gather(
            self.codebooks,
            dim=1,
            index=codes.expand(-1, -1, -1, self.sub_dim)
        )
        
        # Reshape to (num_embeddings, embedding_dim)
        return embeddings.view(self.num_embeddings, self.embedding_dim)

    def forward(
        self,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Input token ids

        Returns:
            Embeddings
        """
        # Get full embeddings
        embeddings = self._get_embeddings()
        
        # Lookup embeddings
        return F.embedding(
            input_ids,
            embeddings,
            padding_idx=self.padding_idx
        )

    def get_codebook_usage(self) -> Dict[int, int]:
        """
        Get codebook usage statistics.

        Returns:
            Dictionary mapping code indices to usage counts
        """
        usage = defaultdict(int)
        for i in range(self.m):
            for j in range(self.k):
                usage[j] += (self.codes[:, i] == j).sum().item()
        return dict(usage)

    def prune_unused_codes(self, min_usage: int = 1):
        """
        Prune unused codes.

        Args:
            min_usage: Minimum usage count
        """
        usage = self.get_codebook_usage()
        unused = {k for k, v in usage.items() if v < min_usage}
        
        if not unused:
            return
        
        # Update codes
        for i in range(self.m):
            mask = torch.isin(self.codes[:, i], torch.tensor(list(unused)))
            if mask.any():
                # Replace unused codes with random codes
                new_codes = torch.randint(
                    0, self.k,
                    (mask.sum(),),
                    device=self.codes.device
                )
                self.codes[mask, i] = new_codes 