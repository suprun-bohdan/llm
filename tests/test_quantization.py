"""
Тести для квантизації моделі.
"""
import pytest
import torch
import torch.nn as nn
from model.quantization import (
    QuantizedLinear,
    DynamicQuantizer,
    QuantizationAwareTraining
)


@pytest.fixture
def linear_layer():
    """Фікстура для лінійного шару."""
    return nn.Linear(10, 20)


@pytest.fixture
def quantized_linear(linear_layer):
    """Фікстура для квантизованого лінійного шару."""
    return QuantizedLinear(linear_layer, bits=8)


@pytest.fixture
def simple_model():
    """Фікстура для простої моделі."""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )


@pytest.fixture
def dynamic_quantizer(simple_model):
    """Фікстура для динамічного квантизатора."""
    return DynamicQuantizer(simple_model, bits=8)


@pytest.fixture
def qat(simple_model):
    """Фікстура для квантизації з навчанням."""
    return QuantizationAwareTraining(simple_model, bits=8)


def test_quantized_linear(quantized_linear):
    """Тест квантизованого лінійного шару."""
    x = torch.randn(5, 10)
    
    # Перевірка квантизації
    assert hasattr(quantized_linear, "scale")
    assert hasattr(quantized_linear, "zero_point")
    assert hasattr(quantized_linear, "quantized_weights")
    
    # Перевірка прямого проходу
    out = quantized_linear(x)
    assert out.shape == (5, 20)
    
    # Перевірка, що ваги квантизовані
    weights = quantized_linear.quantized_weights
    assert torch.all(weights >= 0)
    assert torch.all(weights < 2 ** 8)
    
    # Перевірка симетричної квантизації
    quantized_linear_symmetric = QuantizedLinear(
        quantized_linear.original,
        bits=8,
        symmetric=True
    )
    assert torch.all(quantized_linear_symmetric.zero_point == 0)
    
    # Перевірка асиметричної квантизації
    quantized_linear_asymmetric = QuantizedLinear(
        quantized_linear.original,
        bits=8,
        symmetric=False
    )
    assert not torch.all(quantized_linear_asymmetric.zero_point == 0)


def test_dynamic_quantizer(dynamic_quantizer):
    """Тест динамічного квантизатора."""
    x = torch.randn(5, 10)
    
    # Перевірка квантизації моделі
    assert len(dynamic_quantizer.scales) > 0
    assert len(dynamic_quantizer.zero_points) > 0
    
    # Перевірка прямого проходу
    out = dynamic_quantizer.forward(x)
    assert out.shape == (5, 10)
    
    # Перевірка квантизації по каналах
    quantizer_per_channel = DynamicQuantizer(
        dynamic_quantizer.model,
        bits=8,
        per_channel=True
    )
    assert len(quantizer_per_channel.scales) > 0
    
    # Перевірка квантизації тензора
    tensor = torch.randn(10, 20)
    quantized, scale, zero_point = quantizer_per_channel._quantize_tensor(
        tensor,
        "test"
    )
    assert quantized.shape == tensor.shape
    assert scale.shape == (10,)  # По каналах
    assert zero_point.shape == (10,)
    
    # Перевірка деквантизації
    dequantized = quantizer_per_channel._dequantize_tensor(
        quantized,
        "test"
    )
    assert dequantized.shape == tensor.shape


def test_quantization_aware_training(qat):
    """Тест квантизації з навчанням."""
    x = torch.randn(5, 10)
    
    # Перевірка заміни шарів
    for module in qat.model.modules():
        if isinstance(module, QuantizedLinear):
            assert hasattr(module, "scale")
            assert hasattr(module, "zero_point")
    
    # Перевірка підготовки до навчання
    qat.prepare_for_training()
    for module in qat.model.modules():
        if isinstance(module, QuantizedLinear):
            assert module.scale.requires_grad
            assert module.zero_point.requires_grad
    
    # Перевірка підготовки до інференсу
    qat.prepare_for_inference()
    for module in qat.model.modules():
        if isinstance(module, QuantizedLinear):
            assert not module.scale.requires_grad
            assert not module.zero_point.requires_grad
    
    # Перевірка прямого проходу
    out = qat.forward(x)
    assert out.shape == (5, 10)
    
    # Перевірка різних бітності
    qat_4bit = QuantizationAwareTraining(
        qat.model,
        bits=4
    )
    for module in qat_4bit.model.modules():
        if isinstance(module, QuantizedLinear):
            assert module.bits == 4
    
    # Перевірка квантизації по каналах
    qat_per_channel = QuantizationAwareTraining(
        qat.model,
        bits=8,
        per_channel=True
    )
    for module in qat_per_channel.model.modules():
        if isinstance(module, QuantizedLinear):
            assert module.scale.shape == (module.original.out_features,) 