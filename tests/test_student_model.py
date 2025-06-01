import torch
import pytest
from model.student_model import StudentModel

def test_model_initialization():
    vocab_size = 1000
    d_model = 256
    n_layers = 4
    n_heads = 8
    d_ff = 512
    dropout = 0.1
    lora_rank = 4
    distill_alpha = 0.5
    use_lora = True
    ffn_rank = 512
    gradient_checkpointing = False
    mixed_precision = False
    max_seq_len = 128

    model = StudentModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        lora_rank=lora_rank,
        distill_alpha=distill_alpha,
        use_lora=use_lora,
        ffn_rank=ffn_rank,
        gradient_checkpointing=gradient_checkpointing,
        mixed_precision=mixed_precision,
        max_seq_len=max_seq_len,
        activation="gelu"
    )

    assert model.vocab_size == vocab_size
    assert model.d_model == d_model
    assert len(model.layers) == n_layers
    assert model.use_lora == use_lora

def test_forward_pass():
    batch_size = 2
    seq_len = 16
    vocab_size = 1000
    d_model = 256

    model = StudentModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=2,
        n_heads=4,
        d_ff=512,
        dropout=0.1,
        lora_rank=4,
        distill_alpha=0.5,
        use_lora=True,
        ffn_rank=512,
        gradient_checkpointing=False,
        mixed_precision=False,
        max_seq_len=seq_len,
        activation="gelu"
    )

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    output = model(input_ids, attention_mask)
    assert output.shape == (batch_size, seq_len, vocab_size)

def test_different_activations():
    vocab_size = 1000
    d_model = 256
    seq_len = 16
    batch_size = 2

    activations = ["relu", "gelu"]
    
    for activation in activations:
        model = StudentModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=2,
            n_heads=4,
            d_ff=512,
            dropout=0.1,
            lora_rank=4,
            distill_alpha=0.5,
            use_lora=True,
            ffn_rank=512,
            gradient_checkpointing=False,
            mixed_precision=False,
            max_seq_len=seq_len,
            activation=activation
        )

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        output = model(input_ids, attention_mask)
        assert output.shape == (batch_size, seq_len, vocab_size)

def test_lora_adapters():
    vocab_size = 1000
    d_model = 256
    seq_len = 16
    batch_size = 2
    lora_rank = 4

    model = StudentModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=2,
        n_heads=4,
        d_ff=512,
        dropout=0.1,
        lora_rank=lora_rank,
        distill_alpha=0.5,
        use_lora=True,
        ffn_rank=512,
        gradient_checkpointing=False,
        mixed_precision=False,
        max_seq_len=seq_len,
        activation="gelu"
    )

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    output = model(input_ids, attention_mask)
    assert output.shape == (batch_size, seq_len, vocab_size)

    trainable_params = model.get_trainable_params()
    assert len(trainable_params) > 0

    model.freeze_base_model()
    frozen_params = [p for p in model.parameters() if not p.requires_grad]
    assert len(frozen_params) > 0

if __name__ == "__main__":
    test_model_initialization()
    test_forward_pass()
    test_different_activations()
    test_lora_adapters()
    print("Всі тести пройшли успішно!") 