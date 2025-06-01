"""
Model trainer tests.
"""
import os
import pytest
import torch
import tempfile
from torch.utils.data import DataLoader
from model.transformer import TransformerModel
from tokenizer.simple_tokenizer import SimpleTokenizer
from trainer.trainer import Trainer
from data.dataset import TextDataset, create_dataloader


@pytest.fixture
def model_and_tokenizer():
    """Model and tokenizer fixture."""
    vocab_size = 1000
    d_model = 64
    n_heads = 4
    n_layers = 2
    d_ff = 256
    max_seq_len = 32
    
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len
    )
    
    tokenizer = SimpleTokenizer(
        vocab_size=vocab_size,
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
    
    return model, tokenizer


@pytest.fixture
def dataloaders(model_and_tokenizer):
    """Data loaders fixture."""
    model, tokenizer = model_and_tokenizer
    
    train_texts = [
        "привіт світ",
        "як справи",
        "все добре",
        "це тестовий текст",
        "ще один текст"
    ]
    
    val_texts = [
        "привіт світ",
        "як справи"
    ]
    
    train_dataloader = create_dataloader(
        texts=train_texts,
        tokenizer=tokenizer,
        max_seq_len=10,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pad_token_id=tokenizer.token_to_id["<pad>"],
        bos_token_id=tokenizer.token_to_id["<bos>"],
        eos_token_id=tokenizer.token_to_id["<eos>"]
    )
    
    val_dataloader = create_dataloader(
        texts=val_texts,
        tokenizer=tokenizer,
        max_seq_len=10,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pad_token_id=tokenizer.token_to_id["<pad>"],
        bos_token_id=tokenizer.token_to_id["<bos>"],
        eos_token_id=tokenizer.token_to_id["<eos>"]
    )
    
    return train_dataloader, val_dataloader


@pytest.fixture
def trainer(model_and_tokenizer, dataloaders):
    """Trainer fixture."""
    model, tokenizer = model_and_tokenizer
    train_dataloader, val_dataloader = dataloaders
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device="cpu",
        learning_rate=1e-4,
        weight_decay=0.01,
        max_epochs=2,
        save_every=10,
        eval_every=5
    )
    
    return trainer


def test_trainer_init(trainer):
    """Test trainer initialization."""
    assert trainer.model is not None
    assert trainer.tokenizer is not None
    assert trainer.train_dataloader is not None
    assert trainer.val_dataloader is not None
    assert trainer.device == "cpu"
    assert trainer.optimizer is not None
    assert trainer.scheduler is not None
    assert trainer.criterion is not None
    assert trainer.max_epochs == 2
    assert trainer.save_every == 10
    assert trainer.eval_every == 5
    assert trainer.best_val_loss == float("inf")
    assert trainer.epoch == 0
    assert trainer.step == 0
    assert trainer.patience_counter == 0
    assert isinstance(trainer.history, dict)
    assert "train_loss" in trainer.history
    assert "val_loss" in trainer.history
    assert "learning_rates" in trainer.history


def test_train_epoch(trainer):
    """Test training for one epoch."""
    loss = trainer.train_epoch()
    
    assert isinstance(loss, float)
    assert loss > 0
    assert len(trainer.history["train_loss"]) == 1
    assert len(trainer.history["learning_rates"]) == 1
    assert trainer.step > 0


def test_evaluate(trainer):
    """Test validation."""
    loss = trainer.evaluate()
    
    assert isinstance(loss, float)
    assert loss > 0
    assert len(trainer.history["val_loss"]) == 1


def test_train(trainer):
    """Test full training."""
    history = trainer.train()
    
    assert isinstance(history, dict)
    assert "train_loss" in history
    assert "val_loss" in history
    assert "learning_rates" in history
    assert len(history["train_loss"]) == trainer.max_epochs
    assert len(history["val_loss"]) == trainer.max_epochs
    assert len(history["learning_rates"]) == trainer.max_epochs
    assert trainer.epoch == trainer.max_epochs - 1


def test_save_load_checkpoint(trainer, tmp_path):
    """Test checkpoint save/load."""
    trainer.checkpoint_dir = str(tmp_path)
    
    trainer.train_epoch()
    
    filename = "test_checkpoint.pt"
    trainer.save_checkpoint(filename)
    
    path = os.path.join(trainer.checkpoint_dir, filename)
    assert os.path.exists(path)
    
    old_step = trainer.step
    old_epoch = trainer.epoch
    old_loss = trainer.best_val_loss
    
    trainer.step = 0
    trainer.epoch = 0
    trainer.best_val_loss = float("inf")
    
    trainer.load_checkpoint(filename)
    
    assert trainer.step == old_step
    assert trainer.epoch == old_epoch
    assert trainer.best_val_loss == old_loss


def test_generate_text(trainer):
    """Test text generation."""
    trainer.train_epoch()
    
    prompt = "привіт"
    texts = trainer.generate_text(
        prompt=prompt,
        max_length=10,
        strategy="greedy",
        temperature=1.0,
        num_return_sequences=2
    )
    
    assert isinstance(texts, list)
    assert len(texts) == 2
    assert all(isinstance(text, str) for text in texts)
    assert all(len(text) > 0 for text in texts)
    assert all(text.startswith(prompt) for text in texts)


def test_early_stopping(trainer):
    """Test early stopping."""
    trainer.early_stopping_patience = 1
    
    history = trainer.train()
    
    assert len(history["train_loss"]) <= trainer.max_epochs
    assert trainer.patience_counter >= trainer.early_stopping_patience


def test_invalid_checkpoint(trainer):
    """Test loading invalid checkpoint."""
    with pytest.raises(FileNotFoundError):
        trainer.load_checkpoint("non_existent.pt")


def test_device_placement(trainer):
    """Test device placement."""
    assert next(trainer.model.parameters()).device.type == "cpu"
    
    batch = next(iter(trainer.train_dataloader))
    for tensor in batch.values():
        assert tensor.device.type == "cpu" 