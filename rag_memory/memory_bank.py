"""
Memory bank implementation with Faiss IVFPQ.
"""
import torch
import faiss
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import json
import os


class MemoryBank:
    """Memory bank with Faiss IVFPQ index."""

    def __init__(
        self,
        dim: int = 512,
        nlist: int = 100,
        m: int = 8,
        nbits: int = 8,
        device: str = "cuda"
    ):
        """
        Initialize memory bank.

        Args:
            dim: Embedding dimension
            nlist: Number of clusters
            m: Number of subvectors
            nbits: Number of bits per code
            device: Device to use
        """
        self.dim = dim
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.device = device
        
        # Initialize quantizer
        self.quantizer = faiss.IndexFlatL2(dim)
        
        # Initialize index
        self.index = faiss.IndexIVFPQ(
            self.quantizer,
            dim,
            nlist,
            m,
            nbits
        )
        
        # Initialize metadata
        self.metadata = []
        
        # Set device
        if device == "cuda" and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(),
                0,
                self.index
            )

    def build(
        self,
        embeddings: torch.Tensor,
        metadata: List[Dict]
    ):
        """
        Build index.

        Args:
            embeddings: Embedding matrix
            metadata: List of metadata dictionaries
        """
        # Convert embeddings to numpy
        embeddings = embeddings.cpu().numpy().astype(np.float32)
        
        # Train index
        if not self.index.is_trained:
            self.index.train(embeddings)
        
        # Add embeddings
        self.index.add(embeddings)
        
        # Store metadata
        self.metadata = metadata

    def retrieve(
        self,
        query_emb: torch.Tensor,
        topk: int = 5
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Retrieve similar items.

        Args:
            query_emb: Query embedding
            topk: Number of items to retrieve

        Returns:
            Distances and metadata
        """
        # Convert query to numpy
        query_emb = query_emb.cpu().numpy().astype(np.float32)
        
        # Search
        distances, indices = self.index.search(query_emb, topk)
        
        # Get metadata
        metadata = [self.metadata[i] for i in indices[0]]
        
        return torch.tensor(distances[0]), metadata

    def save(
        self,
        index_path: str,
        metadata_path: str
    ):
        """
        Save index and metadata.

        Args:
            index_path: Index save path
            metadata_path: Metadata save path
        """
        # Save index
        if self.device == "cuda":
            index = faiss.index_gpu_to_cpu(self.index)
        else:
            index = self.index
        
        faiss.write_index(index, index_path)
        
        # Save metadata
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f)

    def load(
        self,
        index_path: str,
        metadata_path: str
    ):
        """
        Load index and metadata.

        Args:
            index_path: Index load path
            metadata_path: Metadata load path
        """
        # Load index
        self.index = faiss.read_index(index_path)
        
        # Set device
        if self.device == "cuda" and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(),
                0,
                self.index
            )
        
        # Load metadata
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

    def get_stats(self) -> Dict:
        """
        Get index statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_vectors": self.index.ntotal,
            "is_trained": self.index.is_trained,
            "nlist": self.nlist,
            "m": self.m,
            "nbits": self.nbits,
            "dim": self.dim
        } 