"""
BPE tokenizer implementation.
"""
import json
import re
from collections import Counter, defaultdict
from typing import List, Dict, Set, Tuple, Optional
import numpy as np


class SimpleTokenizer:
    """Simple BPE tokenizer."""

    def __init__(
        self,
        vocab_size: int = 4096,
        min_freq: int = 2,
        special_tokens: Optional[List[str]] = None
    ):
        """
        Initialize tokenizer.

        Args:
            vocab_size: Vocabulary size
            min_freq: Minimum token frequency
            special_tokens: List of special tokens
        """
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.special_tokens = special_tokens or ["<pad>", "<unk>", "<bos>", "<eos>"]
        
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.merges: Dict[Tuple[str, str], str] = {}
        
        self.is_trained = False

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text.

        Args:
            text: Input text

        Returns:
            Processed text
        """
        text = text.lower()
        
        text = re.sub(r"\s+", " ", text).strip()
        
        text = re.sub(r"([.,!?])", r" \1 ", text)
        text = re.sub(r"\s+", " ", text).strip()
        
        return text

    def _get_stats(self, words: List[List[str]]) -> Dict[Tuple[str, str], int]:
        """
        Get character pair statistics.

        Args:
            words: List of tokenized words

        Returns:
            Dictionary of pairs and their frequencies
        """
        pairs = defaultdict(int)
        for word in words:
            for i in range(len(word) - 1):
                pairs[word[i], word[i + 1]] += 1
        return dict(pairs)

    def _merge_pair(
        self,
        pair: Tuple[str, str],
        words: List[List[str]]
    ) -> List[List[str]]:
        """
        Merge character pair in all words.

        Args:
            pair: Character pair
            words: List of tokenized words

        Returns:
            Updated word list
        """
        merged = []
        bigram = "".join(pair)
        replacement = self.merges.get(pair, bigram)
        
        for word in words:
            i = 0
            new_word = []
            while i < len(word):
                if (
                    i < len(word) - 1
                    and word[i] == pair[0]
                    and word[i + 1] == pair[1]
                ):
                    new_word.append(replacement)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            merged.append(new_word)
        
        return merged

    def train(self, texts: List[str]) -> None:
        """
        Train tokenizer on text corpus.

        Args:
            texts: List of training texts
        """
        texts = [self.preprocess_text(text) for text in texts]
        
        words = [list(word) for text in texts for word in text.split()]
        
        word_freqs = Counter(" ".join(word) for word in words)
        
        for token in self.special_tokens:
            self.token_to_id[token] = len(self.token_to_id)
            self.id_to_token[len(self.id_to_token)] = token
        
        for word in words:
            for char in word:
                if char not in self.token_to_id:
                    self.token_to_id[char] = len(self.token_to_id)
                    self.id_to_token[len(self.id_to_token)] = char
        
        num_merges = self.vocab_size - len(self.token_to_id)
        
        for i in range(num_merges):
            pairs = self._get_stats(words)
            if not pairs:
                break
            
            best_pair = max(pairs.items(), key=lambda x: x[1])[0]
            
            merged_token = "".join(best_pair)
            self.merges[best_pair] = merged_token
            
            if merged_token not in self.token_to_id:
                self.token_to_id[merged_token] = len(self.token_to_id)
                self.id_to_token[len(self.id_to_token)] = merged_token
            
            words = self._merge_pair(best_pair, words)
        
        self.is_trained = True

    def encode(self, text: str) -> List[int]:
        """
        Tokenize text.

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        if not self.is_trained:
            raise RuntimeError("Tokenizer is not trained")

        text = self.preprocess_text(text)
        
        words = text.split()
        token_ids = []
        
        for word in words:
            chars = list(word)
            i = 0
            while i < len(chars):
                longest_token = None
                longest_len = 0
                
                for j in range(i + 1, len(chars) + 1):
                    token = "".join(chars[i:j])
                    if token in self.token_to_id:
                        longest_token = token
                        longest_len = j - i
                
                if longest_token is not None:
                    token_ids.append(self.token_to_id[longest_token])
                    i += longest_len
                else:
                    token_ids.append(self.token_to_id["<unk>"])
                    i += 1
        
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Detokenize.

        Args:
            token_ids: List of token IDs

        Returns:
            Text
        """
        if not self.is_trained:
            raise RuntimeError("Tokenizer is not trained")

        tokens = [self.id_to_token.get(id_, "<unk>") for id_ in token_ids]
        
        text = "".join(tokens)
        
        text = text.replace("</w>", " ")
        
        text = re.sub(r"\s+", " ", text).strip()
        
        return text

    def save(self, path: str) -> None:
        """
        Save tokenizer.

        Args:
            path: Path to file
        """
        if not self.is_trained:
            raise RuntimeError("Tokenizer is not trained")

        data = {
            "vocab_size": self.vocab_size,
            "min_freq": self.min_freq,
            "special_tokens": self.special_tokens,
            "token_to_id": self.token_to_id,
            "merges": {f"{k[0]} {k[1]}": v for k, v in self.merges.items()}
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "SimpleTokenizer":
        """
        Load tokenizer.

        Args:
            path: Path to file

        Returns:
            Loaded tokenizer
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls(
            vocab_size=data["vocab_size"],
            min_freq=data["min_freq"],
            special_tokens=data["special_tokens"]
        )
        
        tokenizer.token_to_id = data["token_to_id"]
        tokenizer.id_to_token = {int(k): v for k, v in data["token_to_id"].items()}
        tokenizer.merges = {
            tuple(k.split()): v for k, v in data["merges"].items()
        }
        
        tokenizer.is_trained = True
        
        return tokenizer 