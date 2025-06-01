"""
Квантизація моделі.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
import numpy as np


class QuantizedLinear(nn.Module):
    """Квантизований лінійний шар."""
    
    def __init__(
        self,
        original: nn.Linear,
        bits: int = 8,
        symmetric: bool = True
    ):
        """
        Ініціалізація квантизованого шару.

        Args:
            original: Оригінальний лінійний шар
            bits: Кількість бітів
            symmetric: Симетрична квантизація
        """
        super().__init__()
        self.original = original
        self.bits = bits
        self.symmetric = symmetric
        
        # Параметри квантизації
        self.register_buffer("scale", torch.ones(1))
        self.register_buffer("zero_point", torch.zeros(1))
        
        # Квантизація ваг
        self._quantize_weights()
    
    def _quantize_weights(self):
        """Квантизація ваг."""
        weights = self.original.weight.data
        
        if self.symmetric:
            # Симетрична квантизація
            abs_max = torch.max(torch.abs(weights))
            self.scale = abs_max / (2 ** (self.bits - 1) - 1)
            self.zero_point = torch.zeros_like(self.scale)
        else:
            # Асиметрична квантизація
            min_val = torch.min(weights)
            max_val = torch.max(weights)
            self.scale = (max_val - min_val) / (2 ** self.bits - 1)
            self.zero_point = torch.round(-min_val / self.scale)
        
        # Квантизація
        self.quantized_weights = torch.round(
            weights / self.scale + self.zero_point
        ).clamp(0, 2 ** self.bits - 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямий прохід.

        Args:
            x: Вхідний тензор

        Returns:
            Вихідний тензор
        """
        # Деквантизація ваг
        weights = (self.quantized_weights - self.zero_point) * self.scale
        
        # Лінійне перетворення
        return F.linear(x, weights, self.original.bias)


class DynamicQuantizer:
    """Динамічна квантизація для інференсу."""
    
    def __init__(
        self,
        model: nn.Module,
        bits: int = 8,
        symmetric: bool = True,
        per_channel: bool = False
    ):
        """
        Ініціалізація квантизатора.

        Args:
            model: Модель для квантизації
            bits: Кількість бітів
            symmetric: Симетрична квантизація
            per_channel: Квантизація по каналах
        """
        self.model = model
        self.bits = bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        
        # Параметри квантизації
        self.scales = {}
        self.zero_points = {}
        
        # Квантизація моделі
        self._quantize_model()
    
    def _quantize_tensor(
        self,
        tensor: torch.Tensor,
        name: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Квантизація тензора.

        Args:
            tensor: Тензор для квантизації
            name: Назва тензора

        Returns:
            Кортеж (квантизований тензор, масштаб, нульова точка)
        """
        if self.per_channel:
            # Квантизація по каналах
            if len(tensor.shape) == 2:  # Лінійний шар
                scales = torch.max(torch.abs(tensor), dim=1)[0]
                scales = scales / (2 ** (self.bits - 1) - 1)
                zero_points = torch.zeros_like(scales)
            else:
                raise ValueError("per_channel квантизація підтримується тільки для лінійних шарів")
        else:
            # Квантизація всього тензора
            if self.symmetric:
                abs_max = torch.max(torch.abs(tensor))
                scales = abs_max / (2 ** (self.bits - 1) - 1)
                zero_points = torch.zeros_like(scales)
            else:
                min_val = torch.min(tensor)
                max_val = torch.max(tensor)
                scales = (max_val - min_val) / (2 ** self.bits - 1)
                zero_points = torch.round(-min_val / scales)
        
        # Квантизація
        quantized = torch.round(
            tensor / scales + zero_points
        ).clamp(0, 2 ** self.bits - 1)
        
        return quantized, scales, zero_points
    
    def _quantize_model(self):
        """Квантизація всієї моделі."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Квантизація ваг
                quantized, scale, zero_point = self._quantize_tensor(
                    module.weight.data,
                    f"{name}.weight"
                )
                self.scales[f"{name}.weight"] = scale
                self.zero_points[f"{name}.weight"] = zero_point
                module.weight.data = quantized
                
                # Квантизація зміщення
                if module.bias is not None:
                    quantized, scale, zero_point = self._quantize_tensor(
                        module.bias.data,
                        f"{name}.bias"
                    )
                    self.scales[f"{name}.bias"] = scale
                    self.zero_points[f"{name}.bias"] = zero_point
                    module.bias.data = quantized
    
    def _dequantize_tensor(
        self,
        tensor: torch.Tensor,
        name: str
    ) -> torch.Tensor:
        """
        Деквантизація тензора.

        Args:
            tensor: Квантизований тензор
            name: Назва тензора

        Returns:
            Деквантизований тензор
        """
        scale = self.scales[name]
        zero_point = self.zero_points[name]
        return (tensor - zero_point) * scale
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямий прохід з квантизацією.

        Args:
            x: Вхідний тензор

        Returns:
            Вихідний тензор
        """
        # Квантизація входу
        quantized_x, x_scale, x_zero_point = self._quantize_tensor(x, "input")
        
        # Прямий прохід через модель
        with torch.no_grad():
            out = self.model(quantized_x)
        
        # Деквантизація виходу
        return self._dequantize_tensor(out, "output")


class QuantizationAwareTraining:
    """Квантизація з навчанням."""
    
    def __init__(
        self,
        model: nn.Module,
        bits: int = 8,
        symmetric: bool = True,
        per_channel: bool = False
    ):
        """
        Ініціалізація квантизації з навчанням.

        Args:
            model: Модель для квантизації
            bits: Кількість бітів
            symmetric: Симетрична квантизація
            per_channel: Квантизація по каналах
        """
        self.model = model
        self.bits = bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        
        # Заміна лінійних шарів на квантизовані
        self._replace_layers()
    
    def _replace_layers(self):
        """Заміна лінійних шарів на квантизовані."""
        for name, module in self.model.named_children():
            if isinstance(module, nn.Linear):
                setattr(
                    self.model,
                    name,
                    QuantizedLinear(
                        module,
                        bits=self.bits,
                        symmetric=self.symmetric
                    )
                )
    
    def prepare_for_training(self):
        """Підготовка моделі до навчання."""
        # Встановлення режиму навчання
        self.model.train()
        
        # Встановлення градієнтів для параметрів квантизації
        for module in self.model.modules():
            if isinstance(module, QuantizedLinear):
                module.scale.requires_grad = True
                module.zero_point.requires_grad = True
    
    def prepare_for_inference(self):
        """Підготовка моделі до інференсу."""
        # Встановлення режиму оцінки
        self.model.eval()
        
        # Відключення градієнтів
        for module in self.model.modules():
            if isinstance(module, QuantizedLinear):
                module.scale.requires_grad = False
                module.zero_point.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямий прохід.

        Args:
            x: Вхідний тензор

        Returns:
            Вихідний тензор
        """
        return self.model(x) 