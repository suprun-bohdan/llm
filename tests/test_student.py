import pytest
import torch
import torch.nn as nn
from model.student_model import StudentModel, LoRALinear
from model.distillation import DistillationLoss, DistillationTrainer

@pytest.fixture
def student_config():
    return {
        'd_model': 512,
        'dropout': 0.1,
        'lora_rank': 8,
        'max_seq_len': 512,
        'n_heads': 8,
        'n_layers': 6,
        'd_ff': 2048,
        'distill_alpha': 0.5,
        'use_lora': True,
        'ffn_rank': 2048,
        'gradient_checkpointing': False,
        'mixed_precision': False
    }

@pytest.fixture
def teacher_model():
    model = nn.TransformerEncoder(
        encoder_layer=nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        ),
        num_layers=12
    )
    return model

@pytest.fixture
def student_model(student_config):
    return StudentModel(
        vocab_size=32000,
        **student_config
    )

@pytest.fixture
def distillation_loss():
    return DistillationLoss(
        temperature=2.0,
        alpha=0.5
    )

@pytest.fixture
def distillation_trainer(student_model, teacher_model):
    return DistillationTrainer(
        student=student_model,
        teacher=teacher_model,
        temperature=2.0,
        alpha=0.5,
        device="cpu"  # Використовуємо CPU для тестів
    )

def test_student_model_init(student_model, student_config):
    assert student_model.d_model == student_config["d_model"]
    assert student_model.n_heads == student_config["n_heads"]
    assert len(student_model.layers) == student_config["n_layers"]
    assert student_model.lora_rank == student_config["lora_rank"]
    assert student_model.distill_alpha == student_config["distill_alpha"]

def test_student_model_forward(student_model):
    batch_size = 4
    seq_len = 32
    input_ids = torch.randint(0, 32000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    output = student_model(input_ids, attention_mask)
    assert output.shape == (batch_size, seq_len, 32000)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

def test_lora_adapters(student_model):
    for block in student_model.layers:
        assert isinstance(block.self_attn.out_proj, LoRALinear)
        assert isinstance(block.ffn.w1, LoRALinear)
        assert isinstance(block.ffn.w2, LoRALinear)

def test_distillation_loss(distillation_loss):
    batch_size = 4
    seq_len = 32
    vocab_size = 32000
    
    student_logits = torch.randn(batch_size, seq_len, vocab_size)
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    loss = distillation_loss(student_logits, teacher_logits, labels)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    
    loss_t1 = DistillationLoss(temperature=1.0)(student_logits, teacher_logits, labels)
    loss_t2 = DistillationLoss(temperature=2.0)(student_logits, teacher_logits, labels)
    assert not torch.allclose(loss_t1, loss_t2)
    
    loss_a1 = DistillationLoss(alpha=0.1)(student_logits, teacher_logits, labels)
    loss_a2 = DistillationLoss(alpha=0.9)(student_logits, teacher_logits, labels)
    assert not torch.allclose(loss_a1, loss_a2)

def test_distillation_trainer(distillation_trainer):
    batch_size = 4
    seq_len = 32
    vocab_size = 32000
    
    batch = {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len),
        "target_ids": torch.randint(0, vocab_size, (batch_size, seq_len))
    }
    
    optimizer = torch.optim.AdamW(distillation_trainer.student.parameters())
    metrics = distillation_trainer.train_step(batch, optimizer)
    
    assert isinstance(metrics, dict)
    assert "ce_loss" in metrics
    assert "kl_loss" in metrics
    assert "total_loss" in metrics
    assert not torch.isnan(torch.tensor(metrics["total_loss"]))
    assert not torch.isinf(torch.tensor(metrics["total_loss"]))
    
    eval_metrics = distillation_trainer.evaluate([batch])
    assert isinstance(eval_metrics, dict)
    assert "eval_loss" in eval_metrics
    assert "eval_perplexity" in eval_metrics

def test_model_save_load(student_model, tmp_path):
    path = tmp_path / "student_model.pt"
    torch.save(student_model.state_dict(), str(path))
    
    loaded_model = StudentModel(
        vocab_size=32000,
        d_model=student_model.d_model,
        n_heads=student_model.n_heads,
        n_layers=student_model.n_layers,
        d_ff=student_model.d_ff,
        dropout=student_model.dropout,
        lora_rank=student_model.lora_rank,
        distill_alpha=student_model.distill_alpha,
        use_lora=student_model.use_lora,
        ffn_rank=student_model.d_ff,
        gradient_checkpointing=student_model.gradient_checkpointing,
        mixed_precision=student_model.mixed_precision,
        max_seq_len=student_model.max_seq_len
    )
    loaded_model.load_state_dict(torch.load(str(path)))
    
    assert loaded_model.d_model == student_model.d_model
    assert loaded_model.n_heads == student_model.n_heads
    assert len(loaded_model.layers) == len(student_model.layers)
    assert loaded_model.lora_rank == student_model.lora_rank
    
    for p1, p2 in zip(student_model.parameters(), loaded_model.parameters()):
        assert torch.allclose(p1, p2)

def test_gradient_checkpointing(student_model):
    student_model.gradient_checkpointing = True
    assert student_model.gradient_checkpointing
    
    student_model.gradient_checkpointing = False
    assert not student_model.gradient_checkpointing

def test_mixed_precision(student_model):
    student_model.mixed_precision = True
    assert student_model.mixed_precision
    
    student_model.mixed_precision = False
    assert not student_model.mixed_precision 