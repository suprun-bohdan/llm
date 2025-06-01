"""
Hypernetwork implementation for weight generation.
"""
import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class CoordinateNetwork(nn.Module):
    """Coordinate network for weight generation."""

    def __init__(
        self,
        hidden_sizes: List[int],
        activation: str = "relu",
        output_activation: Optional[str] = None
    ):
        """
        Initialize coordinate network.

        Args:
            hidden_sizes: List of hidden layer sizes
            activation: Activation function name
            output_activation: Output activation function name
        """
        super().__init__()
        
        self.hidden_sizes = hidden_sizes
        self.activation = getattr(nn, activation)()
        self.output_activation = getattr(nn, output_activation)() if output_activation else None
        
        layers = []
        input_size = 3  # (layer_idx, row_idx, col_idx)
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                self.activation
            ])
            input_size = hidden_size
        
        layers.append(nn.Linear(input_size, 1))
        if self.output_activation:
            layers.append(self.output_activation)
        
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        layer_idx: torch.Tensor,
        row_idx: torch.Tensor,
        col_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            layer_idx: Layer indices
            row_idx: Row indices
            col_idx: Column indices

        Returns:
            Generated weights
        """
        x = torch.stack([layer_idx, row_idx, col_idx], dim=-1)
        return self.net(x).squeeze(-1)


class HyperNetwork(nn.Module):
    """Hypernetwork for generating model weights."""

    def __init__(
        self,
        hidden_sizes: List[int],
        layer_shapes: List[Tuple[int, int]],
        activation: str = "relu",
        output_activation: Optional[str] = None
    ):
        """
        Initialize hypernetwork.

        Args:
            hidden_sizes: List of hidden layer sizes
            layer_shapes: List of (rows, cols) for each layer
            activation: Activation function name
            output_activation: Output activation function name
        """
        super().__init__()
        
        self.layer_shapes = layer_shapes
        self.coord_net = CoordinateNetwork(
            hidden_sizes=hidden_sizes,
            activation=activation,
            output_activation=output_activation
        )

    def generate_weights(self, layer_idx: int) -> torch.Tensor:
        """
        Generate weights for a layer.

        Args:
            layer_idx: Layer index

        Returns:
            Generated weight matrix
        """
        rows, cols = self.layer_shapes[layer_idx]
        
        layer_indices = torch.full((rows * cols,), layer_idx, dtype=torch.float)
        row_indices = torch.arange(rows, dtype=torch.float).repeat_interleave(cols)
        col_indices = torch.arange(cols, dtype=torch.float).repeat(rows)
        
        weights = self.coord_net(layer_indices, row_indices, col_indices)
        return weights.view(rows, cols)

    def forward(self) -> List[torch.Tensor]:
        """
        Generate weights for all layers.

        Returns:
            List of generated weight matrices
        """
        return [self.generate_weights(i) for i in range(len(self.layer_shapes))] 