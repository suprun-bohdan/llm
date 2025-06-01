"""
Text generation script with RAG support.
"""
import torch
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Set, Tuple, Union, Any, Callable
import json
import logging
import time
from functools import lru_cache
import warnings
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
import random
import wandb
from torch.cuda.amp import autocast, GradScaler
import yaml
import importlib
import pkg_resources
from concurrent.futures import ThreadPoolExecutor
import asyncio
from abc import ABC, abstractmethod
import torch.distributed as dist
import re
from collections import defaultdict
import os

from student.model_student import StudentModel
from tokenizer.bpe_pq_tokenizer import BPETokenizer
from rag_memory.memory_bank import MemoryBank
from utils.helpers import get_device


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_length: int = 100
    min_length: int = 10
    temperature: float = 1.0
    temp_schedule: Optional[List[float]] = None  # [start, end, decay_steps]
    top_k: int = 50
    top_p: float = 0.9
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 3
    stop_tokens: Set[int] = None
    stop_words: Set[str] = None
    streaming: bool = False
    verbose: bool = False
    num_beams: int = 1
    early_stopping: bool = False
    num_beam_groups: int = 1
    diversity_penalty: float = 0.0
    use_fp16: bool = False
    streaming_throttle: float = 0.0
    batch_size: int = 1
    seed: int = 42
    output_format: str = "text"
    output_path: Optional[str] = None
    dry_run: bool = False
    use_wandb: bool = False
    use_tensorboard: bool = False
    use_ddp: bool = False
    page_size: int = 0
    trim_output: bool = True
    prompt_template: Optional[str] = None


class TokenizerBase(ABC):
    """Base class for tokenizers."""
    
    @abstractmethod
    def encode(self, text: str) -> Dict[str, torch.Tensor]:
        """Encode text to tokens."""
        pass
    
    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """Decode tokens to text."""
        pass
    
    @abstractmethod
    def load(self, path: Path):
        """Load tokenizer from file."""
        pass


class BPETokenizerWrapper(TokenizerBase):
    """Wrapper for BPE tokenizer."""
    
    def __init__(self, tokenizer_class: str = "BPETokenizer"):
        module = importlib.import_module("tokenizer.bpe_pq_tokenizer")
        self.tokenizer = getattr(module, tokenizer_class)()
    
    def encode(self, text: str) -> Dict[str, torch.Tensor]:
        return self.tokenizer.encode(text)
    
    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)
    
    def load(self, path: Path):
        self.tokenizer.load(path)


class ModelLoader:
    """Handles model loading with error handling and device management."""
    
    def __init__(self, model_dir: str, config_path: str, device: str):
        self.model_dir = Path(model_dir)
        self.config_path = Path(config_path)
        self.device = device
        self.logger = logging.getLogger(__name__)
        self._check_dependencies()
        
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
    
    def _check_dependencies(self):
        """Check library versions compatibility."""
        required = {
            "torch": ">=1.8.0",
            "transformers": ">=4.0.0",
            "numpy": ">=1.19.0",
            "faiss-cpu": ">=1.7.0"
        }
        
        for package, version in required.items():
            try:
                pkg_resources.require(f"{package}{version}")
            except pkg_resources.VersionConflict as e:
                warnings.warn(f"Version conflict for {package}: {e}")
            except pkg_resources.DistributionNotFound:
                warnings.warn(f"Package {package} not found")
    
    def load_config(self) -> Dict:
        """Load and validate config."""
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            
            # Validate required fields
            required_fields = {
                "tokenizer": ["vocab_size", "type"],
                "student": ["d_model", "n_heads", "n_layers", "max_seq_len", "dropout"],
                "rag": ["enabled", "dim", "nlist", "m", "nbits"]
            }
            
            for section, fields in required_fields.items():
                if section not in config:
                    raise ValueError(f"Missing section {section} in config")
                for field in fields:
                    if field not in config[section]:
                        raise ValueError(f"Missing field {field} in {section}")
            
            return config
        except yaml.YAMLError:
            raise ValueError(f"Invalid YAML in config file: {self.config_path}")
    
    def load_model(self) -> Tuple[StudentModel, TokenizerBase]:
        """Load model and tokenizer with fallback options."""
        config = self.load_config()
        
        # Load tokenizer
        tokenizer_path = self.model_dir / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
        
        tokenizer = BPETokenizerWrapper(config["tokenizer"]["type"])
        tokenizer.load(tokenizer_path)
        
        # Create model
        model = StudentModel(
            vocab_size=config["tokenizer"]["vocab_size"],
            d_model=config["student"]["d_model"],
            n_heads=config["student"]["n_heads"],
            n_layers=config["student"]["n_layers"],
            d_ff=config["student"]["d_model"] * 4,
            max_seq_len=config["student"]["max_seq_len"],
            dropout=config["student"]["dropout"],
            use_lora=config["student"]["use_lora"],
            lora_rank=config["student"]["lora_rank"]
        )
        
        # Try loading quantized model first
        model_path = self.model_dir / "model_quant.pt"
        if model_path.exists():
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                self.logger.info("Loaded quantized model")
            except Exception as e:
                warnings.warn(f"Failed to load quantized model: {e}. Falling back to full model.")
                model_path = self.model_dir / "best_model.pt"
        else:
            model_path = self.model_dir / "best_model.pt"
        
        if not model_path.exists():
            raise FileNotFoundError(f"No model weights found in {self.model_dir}")
        
        # Load weights
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        
        # Move to device
        try:
            model.to(self.device)
            if self.device == "cpu" and hasattr(model, "quantized"):
                warnings.warn("Running quantized model on CPU may be slow")
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                self.logger.warning("GPU OOM, falling back to CPU")
                self.device = "cpu"
                model.to(self.device)
            else:
                raise
        
        model.eval()
        return model, tokenizer


class RAGStats:
    """Statistics for RAG operations."""
    
    def __init__(self):
        self.total_queries = 0
        self.successful_hits = 0
        self.failed_hits = 0
        self.context_tokens = 0
        self.generated_tokens = 0
        self.rag_context_used = 0
    
    @property
    def hit_rate(self) -> float:
        return self.successful_hits / max(1, self.total_queries)
    
    @property
    def context_usage(self) -> float:
        return self.rag_context_used / max(1, self.total_queries)
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "total_queries": self.total_queries,
            "successful_hits": self.successful_hits,
            "failed_hits": self.failed_hits,
            "hit_rate": self.hit_rate,
            "context_tokens": self.context_tokens,
            "generated_tokens": self.generated_tokens,
            "context_usage": self.context_usage
        }


class TemperatureScheduler:
    """Dynamic temperature scheduling."""
    
    def __init__(self, start: float, end: float, decay_steps: int):
        self.start = start
        self.end = end
        self.decay_steps = decay_steps
    
    def __call__(self, step: int) -> float:
        if step >= self.decay_steps:
            return self.end
        progress = step / self.decay_steps
        return self.start + (self.end - self.start) * progress


class PromptTemplate:
    """Template for prompt formatting."""
    
    def __init__(self, template: str):
        self.template = template
        self.required_fields = set(re.findall(r"{(\w+)}", template))
    
    def format(self, **kwargs) -> str:
        missing = self.required_fields - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        return self.template.format(**kwargs)


class RAGManager:
    """Manages RAG operations with caching and batching."""
    
    def __init__(self, index_path: str, metadata_path: str, config_path: str, device: str, batch_size: int = 32):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.config_path = Path(config_path)
        self.device = device
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
        self.memory_bank = None
        self.stats = RAGStats()
        self._init_memory_bank()
    
    def _init_memory_bank(self):
        """Initialize memory bank if enabled."""
        if not self.index_path.exists():
            raise FileNotFoundError(f"RAG index not found: {self.index_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"RAG metadata not found: {self.metadata_path}")
        
        config = self.load_config()
        if not config["rag"]["enabled"]:
            return
        
        self.memory_bank = MemoryBank(
            dim=config["rag"]["dim"],
            nlist=config["rag"]["nlist"],
            m=config["rag"]["m"],
            nbits=config["rag"]["nbits"],
            device=self.device
        )
        self.memory_bank.load(self.index_path, self.metadata_path)
    
    def load_config(self) -> Dict:
        """Load config."""
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)
    
    @lru_cache(maxsize=100)
    def get_context(self, query_emb: torch.Tensor, topk: int = 3) -> Tuple[str, int]:
        """Get context for query with caching and stats tracking."""
        if self.memory_bank is None:
            return "", 0
        
        self.stats.total_queries += 1
        
        # Process in batches if needed
        if query_emb.size(0) > self.batch_size:
            contexts = []
            total_tokens = 0
            for i in range(0, query_emb.size(0), self.batch_size):
                batch_emb = query_emb[i:i + self.batch_size]
                _, metadata = self.memory_bank.retrieve(batch_emb, topk=topk)
                batch_contexts = ["\n".join(item["text"] for item in m) for m in metadata]
                contexts.extend(batch_contexts)
                total_tokens += sum(len(c.split()) for c in batch_contexts)
            
            context = "\n\n".join(contexts)
        else:
            _, metadata = self.memory_bank.retrieve(query_emb, topk=topk)
            context = "\n".join(item["text"] for item in metadata)
            total_tokens = len(context.split())
        
        if context.strip():
            self.stats.successful_hits += 1
            self.stats.context_tokens += total_tokens
            self.stats.rag_context_used += 1
        else:
            self.stats.failed_hits += 1
        
        return context, total_tokens


class TextGenerator:
    """Handles text generation with advanced features."""
    
    def __init__(
        self,
        model: StudentModel,
        tokenizer: TokenizerBase,
        rag_manager: Optional[RAGManager],
        config: GenerationConfig,
        device: str,
        rank: int = 0,
        world_size: int = 1
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.rag_manager = rag_manager
        self.config = config
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.logger = logging.getLogger(__name__)
        self.scaler = GradScaler() if config.use_fp16 else None
        
        # Set random seeds
        torch.manual_seed(config.seed + rank)
        np.random.seed(config.seed + rank)
        random.seed(config.seed + rank)
        
        # Initialize temperature scheduler
        self.temp_scheduler = None
        if config.temp_schedule:
            self.temp_scheduler = TemperatureScheduler(*config.temp_schedule)
        
        # Initialize prompt template
        self.prompt_template = None
        if config.prompt_template:
            self.prompt_template = PromptTemplate(config.prompt_template)
        
        # Initialize stop tokens and words
        if config.stop_tokens is None:
            config.stop_tokens = {tokenizer.token_to_id["[SEP]"]}
        
        # Setup logging
        if config.use_wandb and rank == 0:
            wandb.init(project="text-generation", config=vars(config))
        if config.use_tensorboard and rank == 0:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir="runs/generation")
    
    def _get_temperature(self, step: int) -> float:
        """Get current temperature value."""
        if self.temp_scheduler:
            return self.temp_scheduler(step)
        return self.config.temperature
    
    def _format_prompt(self, query: str, context: str = "") -> str:
        """Format prompt using template if available."""
        if self.prompt_template:
            return self.prompt_template.format(
                query=query,
                rag_context=context
            )
        if context:
            return f"{context}\n\n{query}"
        return query
    
    def _post_process(self, text: str) -> str:
        """Apply post-processing rules."""
        if not self.config.trim_output:
            return text
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters at start/end
        text = text.strip(' \t\n\r.,;:!?')
        return text
    
    def _check_stop_words(self, text: str) -> bool:
        """Check if text contains any stop words."""
        if not self.config.stop_words:
            return False
        return any(word in text for word in self.config.stop_words)
    
    def _get_next_token(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generated_ngrams: Set[Tuple[int, ...]],
        step: int
    ) -> Tuple[int, float]:
        """Get next token with sampling."""
        with torch.no_grad():
            with autocast(enabled=self.config.use_fp16):
                outputs = self.model(input_ids, attention_mask)
                temperature = self._get_temperature(step)
                next_token_logits = outputs[:, -1, :] / temperature
                
                # Apply length penalty
                if len(input_ids[0]) > self.config.min_length:
                    length_penalty = (len(input_ids[0]) / self.config.max_length) ** self.config.length_penalty
                    next_token_logits = next_token_logits * length_penalty
                
                # Apply top-k filtering
                if self.config.top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, self.config.top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float("-inf")
                
                # Apply nucleus sampling
                if self.config.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > self.config.top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float("-inf")
                
                # Block repeated n-grams
                if self.config.no_repeat_ngram_size > 0:
                    for ngram in generated_ngrams:
                        if len(ngram) == self.config.no_repeat_ngram_size:
                            next_token_logits[0, ngram[-1]] = float("-inf")
                
                # Sample
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Get token probability for logging
                token_prob = probs[0, next_token[0]].item()
                
                return next_token[0].item(), token_prob
    
    def _beam_search(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_beams: int,
        num_beam_groups: int = 1,
        diversity_penalty: float = 0.0
    ) -> List[Tuple[List[int], float]]:
        """Perform beam search generation."""
        batch_size = input_ids.size(0)
        vocab_size = self.model.vocab_size
        
        # Initialize beams
        beams = [(input_ids, attention_mask, 0.0)]  # (input_ids, attention_mask, score)
        finished_beams = []
        
        for step in range(self.config.max_length):
            candidates = []
            
            # Generate next tokens for each beam
            for beam_input_ids, beam_attention_mask, beam_score in beams:
                if len(beam_input_ids[0]) >= self.config.max_length:
                    finished_beams.append((beam_input_ids[0].tolist(), beam_score))
                    continue
                
                next_token, token_prob = self._get_next_token(
                    beam_input_ids,
                    beam_attention_mask,
                    set(),  # No n-gram blocking in beam search
                    step
                )
                
                # Update beam
                new_input_ids = torch.cat([
                    beam_input_ids,
                    torch.tensor([[next_token]], device=self.device)
                ], dim=1)
                new_attention_mask = torch.cat([
                    beam_attention_mask,
                    torch.ones(1, 1, device=self.device)
                ], dim=1)
                new_score = beam_score - torch.log(torch.tensor(token_prob))
                
                candidates.append((new_input_ids, new_attention_mask, new_score))
                
                # Check if beam is finished
                if next_token in self.config.stop_tokens:
                    finished_beams.append((new_input_ids[0].tolist(), new_score))
            
            # Select top beams
            candidates.sort(key=lambda x: x[2])
            beams = candidates[:num_beams]
            
            # Apply diversity penalty if using multiple beam groups
            if num_beam_groups > 1:
                for i in range(num_beam_groups):
                    group_beams = beams[i::num_beam_groups]
                    for j, (input_ids, attention_mask, score) in enumerate(group_beams):
                        for k, (other_ids, _, _) in enumerate(group_beams):
                            if j != k:
                                # Add penalty based on token overlap
                                overlap = len(set(input_ids[0].tolist()) & set(other_ids[0].tolist()))
                                score += diversity_penalty * overlap
        
        # Add remaining beams to finished
        finished_beams.extend([(b[0].tolist(), b[2]) for b in beams])
        
        # Sort by score and return top beams
        finished_beams.sort(key=lambda x: x[1])
        return finished_beams[:num_beams]
    
    def _paginate_text(self, text: str, page_size: int) -> List[str]:
        """Split text into pages of specified size."""
        if not page_size:
            return [text]
        
        tokens = text.split()
        pages = []
        for i in range(0, len(tokens), page_size):
            page = " ".join(tokens[i:i + page_size])
            pages.append(page)
        return pages
    
    def generate(
        self,
        prompts: Union[str, List[str]],
        callback=None
    ) -> Union[Tuple[str, Dict[str, float]], List[Tuple[str, Dict[str, float]]]]:
        """Generate text with streaming and metrics."""
        if isinstance(prompts, str):
            prompts = [prompts]
        
        # Distribute prompts across GPUs
        if self.config.use_ddp:
            prompts = prompts[self.rank::self.world_size]
        
        results = []
        for prompt in prompts:
            start_time = time.time()
            
            # Get context from RAG if available
            context = ""
            context_tokens = 0
            if self.rag_manager is not None:
                with torch.no_grad():
                    inputs = self.tokenizer.encode(prompt)
                    input_ids = inputs["input_ids"].unsqueeze(0).to(self.device)
                    attention_mask = inputs["attention_mask"].unsqueeze(0).to(self.device)
                    
                    outputs = self.model(input_ids, attention_mask)
                    query_emb = outputs.mean(dim=1)
                    context, context_tokens = self.rag_manager.get_context(query_emb)
            
            # Format prompt
            formatted_prompt = self._format_prompt(prompt, context)
            
            # Encode prompt
            inputs = self.tokenizer.encode(formatted_prompt)
            input_ids = inputs["input_ids"].unsqueeze(0).to(self.device)
            attention_mask = inputs["attention_mask"].unsqueeze(0).to(self.device)
            
            # Check if dry run
            if self.config.dry_run:
                self.logger.info(f"Dry run for prompt: {formatted_prompt}")
                continue
            
            # Generate using beam search or sampling
            if self.config.num_beams > 1:
                beams = self._beam_search(
                    input_ids,
                    attention_mask,
                    self.config.num_beams,
                    self.config.num_beam_groups,
                    self.config.diversity_penalty
                )
                generated = beams[0][0]  # Take best beam
            else:
                # Initialize generation
                generated = []
                generated_ngrams = set()
                metrics = {
                    "num_tokens": 0,
                    "avg_prob": 0.0,
                    "entropy": 0.0,
                    "context_tokens": context_tokens
                }
                
                # Generate
                with torch.no_grad():
                    for step in range(self.config.max_length):
                        # Get next token
                        next_token, token_prob = self._get_next_token(
                            input_ids,
                            attention_mask,
                            generated_ngrams,
                            step
                        )
                        
                        # Update metrics
                        metrics["num_tokens"] += 1
                        metrics["avg_prob"] = (metrics["avg_prob"] * (metrics["num_tokens"] - 1) + token_prob) / metrics["num_tokens"]
                        
                        # Update n-grams
                        if self.config.no_repeat_ngram_size > 0:
                            ngram = tuple(generated[-(self.config.no_repeat_ngram_size-1):] + [next_token])
                            generated_ngrams.add(ngram)
                        
                        # Append token
                        generated.append(next_token)
                        
                        # Update input
                        input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=self.device)], dim=1)
                        attention_mask = torch.cat([
                            attention_mask,
                            torch.ones(1, 1, device=self.device)
                        ], dim=1)
                        
                        # Stream output
                        if self.config.streaming and callback:
                            text = self.tokenizer.decode([next_token])
                            callback(text)
                            if self.config.streaming_throttle > 0:
                                time.sleep(self.config.streaming_throttle)
                        
                        # Check stop conditions
                        if next_token in self.config.stop_tokens:
                            break
                        if len(generated) >= self.config.min_length and len(generated) >= self.config.max_length:
                            break
                        
                        # Check stop words
                        if self.config.stop_words:
                            current_text = self.tokenizer.decode(generated)
                            if self._check_stop_words(current_text):
                                break
            
            # Calculate final metrics
            metrics["time"] = time.time() - start_time
            metrics["tokens_per_second"] = metrics["num_tokens"] / metrics["time"]
            
            # Add RAG stats
            if self.rag_manager:
                metrics.update(self.rag_manager.stats.to_dict())
            
            # Log metrics
            if self.config.verbose and self.rank == 0:
                self.logger.info(f"Generation metrics: {metrics}")
            if self.config.use_wandb and self.rank == 0:
                wandb.log(metrics)
            if self.config.use_tensorboard and self.rank == 0:
                for k, v in metrics.items():
                    self.writer.add_scalar(f"generation/{k}", v)
            
            # Decode and post-process
            text = self.tokenizer.decode(generated)
            text = self._post_process(text)
            
            # Paginate if needed
            if self.config.page_size:
                pages = self._paginate_text(text, self.config.page_size)
                for i, page in enumerate(pages):
                    results.append((page, {**metrics, "page": i + 1}))
            else:
                results.append((text, metrics))
        
        # Gather results from all GPUs
        if self.config.use_ddp:
            all_results = [None for _ in range(self.world_size)]
            dist.all_gather_object(all_results, results)
            if self.rank == 0:
                results = [item for sublist in all_results for item in sublist]
        
        # Save results if output path specified
        if self.config.output_path and self.rank == 0:
            output_path = Path(self.config.output_path)
            if self.config.output_format == "json":
                with open(output_path, "w") as f:
                    json.dump([{
                        "text": text,
                        "metrics": metrics
                    } for text, metrics in results], f, indent=2)
            else:
                with open(output_path, "w") as f:
                    for text, _ in results:
                        f.write(text + "\n\n")
        
        return results[0] if len(results) == 1 else results


def setup_logging(verbose: bool):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def setup_distributed():
    """Setup distributed training."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1
    
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(rank)
    
    return rank, world_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--prompts_file", type=str, help="File with prompts, one per line")
    parser.add_argument("--rag_index", type=str)
    parser.add_argument("--rag_metadata", type=str)
    parser.add_argument("--max_length", type=int, default=100)
    parser.add_argument("--min_length", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temp_schedule", type=float, nargs=3, help="[start, end, decay_steps]")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--stop_tokens", type=str, help="Comma-separated list of stop tokens")
    parser.add_argument("--stop_words_file", type=str, help="File with stop words, one per line")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--streaming_throttle", type=float, default=0.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--num_beam_groups", type=int, default=1)
    parser.add_argument("--diversity_penalty", type=float, default=0.0)
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_format", type=str, choices=["text", "json"], default="text")
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--use_tensorboard", action="store_true")
    parser.add_argument("--use_ddp", action="store_true")
    parser.add_argument("--page_size", type=int, default=0)
    parser.add_argument("--trim_output", action="store_true")
    parser.add_argument("--prompt_template", type=str)
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Setup distributed
        rank, world_size = 0, 1
        if args.use_ddp:
            rank, world_size = setup_distributed()
        
        # Set device
        device = get_device(args.device)
        
        # Load prompts
        prompts = [args.prompt]
        if args.prompts_file:
            with open(args.prompts_file, "r") as f:
                prompts.extend(line.strip() for line in f if line.strip())
        
        # Load stop words
        stop_words = None
        if args.stop_words_file:
            with open(args.stop_words_file, "r") as f:
                stop_words = {line.strip() for line in f if line.strip()}
        
        # Load model
        logger.info("Loading model...")
        model_loader = ModelLoader(args.model_dir, args.config, device)
        model, tokenizer = model_loader.load_model()
        
        # Load RAG if available
        rag_manager = None
        if args.rag_index and args.rag_metadata:
            logger.info("Loading RAG...")
            rag_manager = RAGManager(
                args.rag_index,
                args.rag_metadata,
                args.config,
                device,
                args.batch_size
            )
        
        # Parse stop tokens
        stop_tokens = None
        if args.stop_tokens:
            stop_tokens = {
                tokenizer.token_to_id[t] for t in args.stop_tokens.split(",")
                if t in tokenizer.token_to_id
            }
        
        # Create generator
        config = GenerationConfig(
            max_length=args.max_length,
            min_length=args.min_length,
            temperature=args.temperature,
            temp_schedule=args.temp_schedule,
            top_k=args.top_k,
            top_p=args.top_p,
            length_penalty=args.length_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            stop_tokens=stop_tokens,
            stop_words=stop_words,
            streaming=args.streaming,
            streaming_throttle=args.streaming_throttle,
            verbose=args.verbose,
            num_beams=args.num_beams,
            early_stopping=args.early_stopping,
            num_beam_groups=args.num_beam_groups,
            diversity_penalty=args.diversity_penalty,
            use_fp16=args.use_fp16,
            batch_size=args.batch_size,
            seed=args.seed,
            output_format=args.output_format,
            output_path=args.output_path,
            dry_run=args.dry_run,
            use_wandb=args.use_wandb,
            use_tensorboard=args.use_tensorboard,
            use_ddp=args.use_ddp,
            page_size=args.page_size,
            trim_output=args.trim_output,
            prompt_template=args.prompt_template
        )
        
        generator = TextGenerator(
            model,
            tokenizer,
            rag_manager,
            config,
            device,
            rank,
            world_size
        )
        
        # Generate text
        logger.info("Generating text...")
        
        def stream_callback(text: str):
            print(text, end="", flush=True)
        
        results = generator.generate(
            prompts,
            callback=stream_callback if args.streaming else None
        )
        
        if not args.streaming and rank == 0:
            if len(results) == 1:
                text, metrics = results[0]
                print("\nGenerated text:")
                print(text)
                if args.verbose:
                    print("\nMetrics:")
                    for k, v in metrics.items():
                        print(f"{k}: {v:.4f}")
            else:
                print("\nGenerated texts:")
                for i, (text, metrics) in enumerate(results, 1):
                    print(f"\nText {i}:")
                    print(text)
                    if args.verbose:
                        print("Metrics:")
                        for k, v in metrics.items():
                            print(f"{k}: {v:.4f}")
    
    except Exception as e:
        logger.error(f"Error during generation: {e}", exc_info=args.verbose)
        raise
    finally:
        if args.use_wandb and rank == 0:
            wandb.finish()
        if args.use_tensorboard and rank == 0:
            generator.writer.close()
        if args.use_ddp:
            dist.destroy_process_group()


if __name__ == "__main__":
    main() 