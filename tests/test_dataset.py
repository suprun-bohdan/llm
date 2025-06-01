"""
Dataset tests.
"""
import pytest
import torch
import tempfile
import json
from data.dataset import (
    TextDataset,
    create_dataloader,
    load_jsonl_dataset,
    split_dataset,
    collate_fn
)
from tokenizer.simple_tokenizer import SimpleTokenizer


@pytest.fixture
def tokenizer():
    """Tokenizer fixture."""
    tokenizer = SimpleTokenizer(
        vocab_size=100,
        min_freq=2,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"]
    )
    
    texts = [
        "привіт світ",
        "привіт світ",
        "як справи",
        "як справи",
        "все добре",
        "все добре"
    ]
    tokenizer.train(texts)
    
    return tokenizer


@pytest.fixture
def texts():
    """Texts fixture."""
    return [
        "привіт світ",
        "як справи",
        "все добре",
        "це тестовий текст для перевірки роботи набору даних",
        "ще один текст для тестування"
    ]


def test_text_dataset(tokenizer, texts):
    """Test dataset."""
    dataset = TextDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_seq_len=10,
        pad_token_id=tokenizer.token_to_id["<pad>"],
        bos_token_id=tokenizer.token_to_id["<bos>"],
        eos_token_id=tokenizer.token_to_id["<eos>"]
    )
    
    assert len(dataset) == len(texts)
    
    item = dataset[0]
    assert isinstance(item, dict)
    assert "input_ids" in item
    assert "target_ids" in item
    assert "attention_mask" in item
    
    assert item["input_ids"].shape == (9,)
    assert item["target_ids"].shape == (9,)
    assert item["attention_mask"].shape == (9,)
    
    assert item["input_ids"].dtype == torch.long
    assert item["target_ids"].dtype == torch.long
    assert item["attention_mask"].dtype == torch.float
    
    assert item["input_ids"][0] == tokenizer.token_to_id["<bos>"]
    assert item["target_ids"][-1] == tokenizer.token_to_id["<eos>"]
    
    long_text = "це дуже довгий текст який повинен бути обрізаний до максимальної довжини послідовності"
    item = dataset[3]
    assert len(item["input_ids"]) == 9
    assert item["attention_mask"][-1] == 0


def test_create_dataloader(tokenizer, texts):
    """Test dataloader creation."""
    dataloader = create_dataloader(
        texts=texts,
        tokenizer=tokenizer,
        max_seq_len=10,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pad_token_id=tokenizer.token_to_id["<pad>"],
        bos_token_id=tokenizer.token_to_id["<bos>"],
        eos_token_id=tokenizer.token_to_id["<eos>"]
    )
    
    batch = next(iter(dataloader))
    assert isinstance(batch, dict)
    assert "input_ids" in batch
    assert "target_ids" in batch
    assert "attention_mask" in batch
    
    assert batch["input_ids"].shape == (2, 9)
    assert batch["target_ids"].shape == (2, 9)
    assert batch["attention_mask"].shape == (2, 9)
    
    assert batch["input_ids"].dtype == torch.long
    assert batch["target_ids"].dtype == torch.long
    assert batch["attention_mask"].dtype == torch.float


def test_load_jsonl_dataset():
    """Test loading data from JSONL file."""
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as f:
        data = [
            {"text": "перший текст", "meta": "meta1"},
            {"text": "другий текст", "meta": "meta2"},
            {"text": "третій текст", "meta": "meta3"}
        ]
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        temp_path = f.name
    
    try:
        texts = load_jsonl_dataset(temp_path)
        assert len(texts) == 3
        assert texts[0] == "перший текст"
        
        texts = load_jsonl_dataset(temp_path, max_samples=2)
        assert len(texts) == 2
        
        texts = load_jsonl_dataset(temp_path, text_field="meta")
        assert texts[0] == "meta1"
        
    finally:
        import os
        os.unlink(temp_path)


def test_split_dataset(texts):
    """Test dataset splitting."""
    train, val, test = split_dataset(
        texts=texts,
        val_split=0.2,
        test_split=0.2
    )
    
    n = len(texts)
    assert len(train) == int(n * 0.6)
    assert len(val) == int(n * 0.2)
    assert len(test) == int(n * 0.2)
    
    train1, val1, test1 = split_dataset(
        texts=texts,
        val_split=0.2,
        test_split=0.2,
        seed=42
    )
    train2, val2, test2 = split_dataset(
        texts=texts,
        val_split=0.2,
        test_split=0.2,
        seed=42
    )
    
    assert train1 == train2
    assert val1 == val2
    assert test1 == test2


def test_collate_fn(tokenizer, texts):
    """Test batch collation function."""
    dataset = TextDataset(
        texts=texts[:2],
        tokenizer=tokenizer,
        max_seq_len=10,
        pad_token_id=tokenizer.token_to_id["<pad>"],
        bos_token_id=tokenizer.token_to_id["<bos>"],
        eos_token_id=tokenizer.token_to_id["<eos>"]
    )
    
    batch = [dataset[i] for i in range(2)]
    collated = collate_fn(batch)
    
    assert isinstance(collated, dict)
    assert "input_ids" in collated
    assert "target_ids" in collated
    assert "attention_mask" in collated
    
    assert collated["input_ids"].shape == (2, 9)
    assert collated["target_ids"].shape == (2, 9)
    assert collated["attention_mask"].shape == (2, 9)
    
    assert collated["input_ids"].dtype == torch.long
    assert collated["target_ids"].dtype == torch.long
    assert collated["attention_mask"].dtype == torch.float 