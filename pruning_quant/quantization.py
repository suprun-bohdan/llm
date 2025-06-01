"""
Model quantization implementation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class Quantizer:
    """Base quantizer class."""

    def __init__(
        self,
        bits: int = 8,
        symmetric: bool = True,
        per_channel: bool = False
    ):
        """
        Initialize quantizer.

        Args:
            bits: Number of bits for quantization
            symmetric: Whether to use symmetric quantization
            per_channel: Whether to quantize per channel
        """
        self.bits = bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        
        self.scale = None
        self.zero_point = None

    def _get_quantization_range(self) -> Tuple[float, float]:
        """
        Get quantization range.

        Returns:
            Min and max values for quantization
        """
        if self.symmetric:
            max_val = 2 ** (self.bits - 1) - 1
            return -max_val, max_val
        else:
            return 0, 2 ** self.bits - 1

    def _compute_scale_zero_point(
        self,
        tensor: torch.Tensor,
        min_val: float,
        max_val: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scale and zero point.

        Args:
            tensor: Input tensor
            min_val: Minimum value
            max_val: Maximum value

        Returns:
            Scale and zero point tensors
        """
        if self.per_channel:
            if len(tensor.shape) == 2:  # Linear layer
                dim = 1
            else:  # Conv layer
                dim = 0
                
            min_vals = tensor.min(dim=dim)[0]
            max_vals = tensor.max(dim=dim)[0]
        else:
            min_vals = torch.tensor(tensor.min().item())
            max_vals = torch.tensor(tensor.max().item())
        
        qmin, qmax = self._get_quantization_range()
        
        scale = (max_vals - min_vals) / (qmax - qmin)
        scale = torch.clamp(scale, min=1e-8)
        
        if self.symmetric:
            zero_point = torch.zeros_like(scale)
        else:
            zero_point = qmin - min_vals / scale
            zero_point = torch.clamp(zero_point, qmin, qmax)
            zero_point = zero_point.round()
        
        return scale, zero_point

    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Quantize tensor.

        Args:
            tensor: Input tensor

        Returns:
            Quantized tensor
        """
        if self.scale is None or self.zero_point is None:
            min_val, max_val = self._get_quantization_range()
            self.scale, self.zero_point = self._compute_scale_zero_point(
                tensor, min_val, max_val
            )
        
        if self.per_channel:
            if len(tensor.shape) == 2:  # Linear layer
                scale = self.scale.unsqueeze(0)
                zero_point = self.zero_point.unsqueeze(0)
            else:  # Conv layer
                scale = self.scale.view(-1, 1, 1, 1)
                zero_point = self.zero_point.view(-1, 1, 1, 1)
        else:
            scale = self.scale
            zero_point = self.zero_point
        
        qmin, qmax = self._get_quantization_range()
        
        # Quantize
        tensor = tensor / scale + zero_point
        tensor = torch.clamp(tensor, qmin, qmax)
        tensor = tensor.round()
        
        # Dequantize
        tensor = (tensor - zero_point) * scale
        
        return tensor


class DynamicQuantizer(Quantizer):
    """Dynamic quantization."""

    def __init__(
        self,
        bits: int = 8,
        symmetric: bool = True,
        per_channel: bool = False
    ):
        """
        Initialize dynamic quantizer.

        Args:
            bits: Number of bits for quantization
            symmetric: Whether to use symmetric quantization
            per_channel: Whether to quantize per channel
        """
        super().__init__(bits, symmetric, per_channel)

    def quantize_linear(
        self,
        module: nn.Linear,
        input_scale: Optional[torch.Tensor] = None
    ) -> nn.Linear:
        """
        Quantize linear layer.

        Args:
            module: Linear layer
            input_scale: Input scale (optional)

        Returns:
            Quantized linear layer
        """
        # Quantize weights
        weight = module.weight.data
        if input_scale is not None:
            weight = weight * input_scale
        
        weight = self.quantize(weight)
        module.weight.data = weight
        
        return module

    def quantize_conv2d(
        self,
        module: nn.Conv2d,
        input_scale: Optional[torch.Tensor] = None
    ) -> nn.Conv2d:
        """
        Quantize Conv2d layer.

        Args:
            module: Conv2d layer
            input_scale: Input scale (optional)

        Returns:
            Quantized Conv2d layer
        """
        # Quantize weights
        weight = module.weight.data
        if input_scale is not None:
            weight = weight * input_scale.view(-1, 1, 1, 1)
        
        weight = self.quantize(weight)
        module.weight.data = weight
        
        return module


class StaticQuantizer(Quantizer):
    """Static quantization."""

    def __init__(
        self,
        bits: int = 8,
        symmetric: bool = True,
        per_channel: bool = False,
        calibration_data: Optional[List[torch.Tensor]] = None
    ):
        """
        Initialize static quantizer.

        Args:
            bits: Number of bits for quantization
            symmetric: Whether to use symmetric quantization
            per_channel: Whether to quantize per channel
            calibration_data: Calibration data for computing scales
        """
        super().__init__(bits, symmetric, per_channel)
        self.calibration_data = calibration_data
        self.activation_scales = {}

    def calibrate(self, model: nn.Module):
        """
        Calibrate model.

        Args:
            model: Model to calibrate
        """
        if self.calibration_data is None:
            return
        
        model.eval()
        
        # Collect activation statistics
        activation_stats = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    activation_stats[name] = output.detach()
            return hook
        
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # Forward pass with calibration data
        with torch.no_grad():
            for data in self.calibration_data:
                model(data)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Compute activation scales
        for name, activations in activation_stats.items():
            min_val, max_val = self._get_quantization_range()
            scale, zero_point = self._compute_scale_zero_point(
                activations, min_val, max_val
            )
            self.activation_scales[name] = scale

    def quantize_model(self, model: nn.Module) -> nn.Module:
        """
        Quantize model.

        Args:
            model: Model to quantize

        Returns:
            Quantized model
        """
        # Calibrate if needed
        if not self.activation_scales:
            self.calibrate(model)
        
        # Quantize layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                input_scale = self.activation_scales.get(name, None)
                self.quantize_linear(module, input_scale)
            elif isinstance(module, nn.Conv2d):
                input_scale = self.activation_scales.get(name, None)
                self.quantize_conv2d(module, input_scale)
        
        return model 