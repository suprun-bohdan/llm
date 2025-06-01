"""
Model tests.
"""
import pytest
import torch
from model.transformer import (
    TransformerModel,
    PositionalEncoding,
    MultiHeadAttention,
    FeedForward,
    TransformerBlock
)


@pytest.fixture
def model():
    """Model fixture."""
    return TransformerModel(
        vocab_size=1000,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        max_seq_len=128,
        dropout=0.1
    )


def test_model_init(model):
    """Test model initialization."""
    assert model.embedding.num_embeddings == 1000
    assert model.embedding.embedding_dim == 64
    assert len(model.blocks) == 2
    assert model.output.out_features == 1000


def test_positional_encoding():
    """Test positional encoding."""
    d_model = 64
    max_seq_len = 128
    pe = PositionalEncoding(d_model, max_seq_len)
    
    x = torch.randn(1, max_seq_len, d_model)
    output = pe(x)
    
    assert output.shape == (1, max_seq_len, d_model)
    assert not torch.allclose(output, x)


def test_multi_head_attention():
    """Test multi-head attention."""
    batch_size = 2
    seq_len = 10
    d_model = 64
    n_heads = 4
    
    attn = MultiHeadAttention(d_model, n_heads)
    q = torch.randn(batch_size, seq_len, d_model)
    k = torch.randn(batch_size, seq_len, d_model)
    v = torch.randn(batch_size, seq_len, d_model)
    
    output = attn(q, k, v)
    assert output.shape == (batch_size, seq_len, d_model)


def test_feed_forward():
    """Test feed-forward network."""
    batch_size = 2
    seq_len = 10
    d_model = 64
    d_ff = 256
    
    ff = FeedForward(d_model, d_ff)
    x = torch.randn(batch_size, seq_len, d_model)
    
    output = ff(x)
    assert output.shape == (batch_size, seq_len, d_model)


def test_transformer_block():
    """Test transformer block."""
    batch_size = 2
    seq_len = 10
    d_model = 64
    n_heads = 4
    d_ff = 256
    
    block = TransformerBlock(d_model, n_heads, d_ff)
    x = torch.randn(batch_size, seq_len, d_model)
    
    output = block(x)
    assert output.shape == (batch_size, seq_len, d_model)


def test_model_forward(model):
    """Test model forward pass."""
    batch_size = 2
    seq_len = 10
    
    x = torch.randint(0, 1000, (batch_size, seq_len))
    output = model(x)
    
    assert output.shape == (batch_size, seq_len, 1000)


def test_model_generate(model):
    """Test text generation."""
    pass


def test_model_save_load(model, tmp_path):
    """Test model save/load."""
    path = tmp_path / "model.pth"
    model.save(str(path))
    
    loaded_model = TransformerModel.load(
        str(path),
        vocab_size=1000,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        max_seq_len=128,
        dropout=0.1
    )
    
    for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
        assert torch.allclose(p1, p2) 