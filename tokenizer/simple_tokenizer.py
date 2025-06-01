"""
Реалізація BPE токенізатора.
"""
import json
import re
from collections import Counter, defaultdict
from typing import List, Dict, Set, Tuple, Optional
import numpy as np


class SimpleTokenizer:
    """Простий BPE токенізатор."""

    def __init__(
        self,
        vocab_size: int = 4096,
        min_freq: int = 2,
        special_tokens: Optional[List[str]] = None
    ):
        """
        Ініціалізація токенізатора.

        Args:
            vocab_size: Розмір словника
            min_freq: Мінімальна частота для токена
            special_tokens: Список спеціальних токенів
        """
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.special_tokens = special_tokens or ["<pad>", "<unk>", "<bos>", "<eos>"]
        
        # Словники
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.merges: Dict[Tuple[str, str], str] = {}
        
        # Флаги
        self.is_trained = False

    def preprocess_text(self, text: str) -> str:
        """
        Попередня обробка тексту.

        Args:
            text: Вхідний текст

        Returns:
            Оброблений текст
        """
        # Приведення до нижнього регістру
        text = text.lower()
        
        # Видалення зайвих пробілів
        text = re.sub(r"\s+", " ", text).strip()
        
        # Додавання пробілів навколо пунктуації
        text = re.sub(r"([.,!?])", r" \1 ", text)
        text = re.sub(r"\s+", " ", text).strip()
        
        return text

    def _get_stats(self, words: List[List[str]]) -> Dict[Tuple[str, str], int]:
        """
        Отримання статистики пар символів.

        Args:
            words: Список токенізованих слів

        Returns:
            Словник пар та їх частот
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
        Об'єднання пари символів у всіх словах.

        Args:
            pair: Пара символів
            words: Список токенізованих слів

        Returns:
            Оновлений список слів
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
        Навчання токенізатора на корпусі текстів.

        Args:
            texts: Список текстів для навчання
        """
        # Попередня обробка
        texts = [self.preprocess_text(text) for text in texts]
        
        # Токенізація на символи
        words = [list(word) for text in texts for word in text.split()]
        
        # Підрахунок частот слів
        word_freqs = Counter(" ".join(word) for word in words)
        
        # Додавання спеціальних токенів
        for token in self.special_tokens:
            self.token_to_id[token] = len(self.token_to_id)
            self.id_to_token[len(self.id_to_token)] = token
        
        # Додавання базових символів
        for word in words:
            for char in word:
                if char not in self.token_to_id:
                    self.token_to_id[char] = len(self.token_to_id)
                    self.id_to_token[len(self.id_to_token)] = char
        
        # BPE навчання
        num_merges = self.vocab_size - len(self.token_to_id)
        
        for i in range(num_merges):
            # Отримання статистики
            pairs = self._get_stats(words)
            if not pairs:
                break
            
            # Вибір найкращої пари
            best_pair = max(pairs.items(), key=lambda x: x[1])[0]
            
            # Об'єднання пари
            merged_token = "".join(best_pair)
            self.merges[best_pair] = merged_token
            
            # Додавання нового токена
            if merged_token not in self.token_to_id:
                self.token_to_id[merged_token] = len(self.token_to_id)
                self.id_to_token[len(self.id_to_token)] = merged_token
            
            # Оновлення слів
            words = self._merge_pair(best_pair, words)
        
        self.is_trained = True

    def encode(self, text: str) -> List[int]:
        """
        Токенізація тексту.

        Args:
            text: Вхідний текст

        Returns:
            Список ID токенів
        """
        if not self.is_trained:
            raise RuntimeError("Токенізатор не навчений")

        # Попередня обробка
        text = self.preprocess_text(text)
        
        # Токенізація на слова
        words = text.split()
        token_ids = []
        
        for word in words:
            # Токенізація слова
            chars = list(word)
            i = 0
            while i < len(chars):
                # Пошук найдовшого токена
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
                    # Невідомий символ
                    token_ids.append(self.token_to_id["<unk>"])
                    i += 1
        
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Детокенізація.

        Args:
            token_ids: Список ID токенів

        Returns:
            Текст
        """
        if not self.is_trained:
            raise RuntimeError("Токенізатор не навчений")

        # Конвертація ID в токени
        tokens = [self.id_to_token.get(id_, "<unk>") for id_ in token_ids]
        
        # Об'єднання токенів
        text = "".join(tokens)
        
        # Відновлення пробілів
        text = text.replace("</w>", " ")
        
        # Видалення зайвих пробілів
        text = re.sub(r"\s+", " ", text).strip()
        
        return text

    def save(self, path: str) -> None:
        """
        Збереження токенізатора.

        Args:
            path: Шлях до файлу
        """
        if not self.is_trained:
            raise RuntimeError("Токенізатор не навчений")

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
        Завантаження токенізатора.

        Args:
            path: Шлях до файлу

        Returns:
            Завантажений токенізатор
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