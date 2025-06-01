"""
Model optimizations tests.
"""
import pytest
import torch
import torch.nn as nn
from model.optimizations import (
    ReversibleBlock,
    LowRankLinear,
    LoRALayer,
    PerformerAttention,
    MagnitudePruner,
    MemoryBank
)


@pytest.fixture
def reversible_block():
    """Reversible block fixture."""
    f = nn.Linear(10, 10)
    g = nn.Linear(10, 10)
    return ReversibleBlock(f, g)


@pytest.fixture
def low_rank_linear():
    """Low-rank linear layer fixture."""
    return LowRankLinear(10, 20, rank=5)


@pytest.fixture
def lora_layer():
    """LoRA layer fixture."""
    original = nn.Linear(10, 20)
    return LoRALayer(original, rank=4)


@pytest.fixture
def performer_attention():
    """Performer attention fixture."""
    return PerformerAttention(d_model=64, n_heads=4)


@pytest.fixture
def simple_model():
    """Simple model fixture for pruning."""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )
    return model


@pytest.fixture
def memory_bank():
    """Memory bank fixture."""
    return MemoryBank(dim=10)


def test_reversible_block(reversible_block):
    """Test reversible block."""
    x1 = torch.randn(5, 10)
    x2 = torch.randn(5, 10)
    
    y1, y2 = reversible_block(x1, x2)
    assert y1.shape == x1.shape
    assert y2.shape == x2.shape
    
    x1_back, x2_back = reversible_block.backward(y1, y2)
    assert torch.allclose(x1, x1_back, atol=1e-6)
    assert torch.allclose(x2, x2_back, atol=1e-6)


def test_low_rank_linear(low_rank_linear):
    """Test low-rank linear layer."""
    x = torch.randn(5, 10)
    out = low_rank_linear(x)
    
    assert out.shape == (5, 20)
    assert low_rank_linear.u.weight.shape == (5, 10)
    assert low_rank_linear.v.weight.shape == (20, 5)


def test_lora_layer(lora_layer):
    """Test LoRA layer."""
    x = torch.randn(5, 10)
    out = lora_layer(x)
    
    assert out.shape == (5, 20)
    assert lora_layer.lora_a.weight.shape == (4, 10)
    assert lora_layer.lora_b.weight.shape == (20, 4)
    
    original_out = lora_layer.original(x)
    assert not torch.allclose(out, original_out)


def test_performer_attention(performer_attention):
    """Test Performer attention."""
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, 64)
    mask = torch.ones(batch_size, seq_len, seq_len)
    
    out = performer_attention(x, mask)
    assert out.shape == (batch_size, seq_len, 64)
    
    mask[0, 0, 1:] = 0
    out_masked = performer_attention(x, mask)
    assert not torch.allclose(out, out_masked)


def test_magnitude_pruner(simple_model):
    """Test pruner."""
    pruner = MagnitudePruner(
        simple_model,
        amount=0.5,
        schedule="gradual",
        start_epoch=0,
        end_epoch=10
    )
    
    for name, mask in pruner.masks.items():
        assert torch.all(mask == 1.0)
    
    pruner.step(epoch=5)
    
    for name, module in simple_model.named_modules():
        if isinstance(module, nn.Linear):
            assert torch.sum(module.weight == 0) > 0


def test_memory_bank(memory_bank):
    """Test memory bank."""
    for i in range(5):
        vector = torch.randn(10)
        metadata = {"id": i}
        memory_bank.add(vector, metadata)
    
    assert len(memory_bank.vectors) == 5
    
    query = torch.randn(10)
    results = memory_bank.retrieve(query, top_k=3)
    
    assert len(results) == 3
    for vector, metadata, similarity in results:
        assert vector.shape == (10,)
        assert isinstance(metadata, dict)
        assert isinstance(similarity, float)
    
    for i in range(10000):
        vector = torch.randn(10)
        metadata = {"id": i}
        memory_bank.add(vector, metadata)
    
    assert len(memory_bank.vectors) == memory_bank.max_size
    
    memory_bank.save("test_memory.pt")
    new_bank = MemoryBank(dim=10)
    new_bank.load("test_memory.pt")
    
    assert len(new_bank.vectors) == len(memory_bank.vectors)
    assert torch.allclose(
        torch.stack(new_bank.vectors),
        torch.stack(memory_bank.vectors)
    ) 