"""
Model pruning implementation.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict


class MagnitudePruner:
    """Magnitude-based pruning."""

    def __init__(
        self,
        model: nn.Module,
        amount: float,
        schedule: str = "gradual",
        start_epoch: int = 0,
        end_epoch: int = 10,
        frequency: int = 1
    ):
        """
        Initialize magnitude pruner.

        Args:
            model: Model to prune
            amount: Target pruning amount (0-1)
            schedule: Pruning schedule ("gradual" or "one-shot")
            start_epoch: Start epoch for gradual pruning
            end_epoch: End epoch for gradual pruning
            frequency: Pruning frequency in epochs
        """
        self.model = model
        self.amount = amount
        self.schedule = schedule
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.frequency = frequency
        
        self.masks = {}
        self._initialize_masks()

    def _initialize_masks(self):
        """Initialize pruning masks."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                self.masks[name] = torch.ones_like(module.weight)

    def _compute_threshold(
        self,
        weights: torch.Tensor,
        amount: float
    ) -> float:
        """
        Compute pruning threshold.

        Args:
            weights: Weight tensor
            amount: Pruning amount

        Returns:
            Pruning threshold
        """
        return torch.quantile(torch.abs(weights), amount)

    def _get_current_amount(self, epoch: int) -> float:
        """
        Get current pruning amount.

        Args:
            epoch: Current epoch

        Returns:
            Current pruning amount
        """
        if self.schedule == "one-shot":
            return self.amount
        
        if epoch < self.start_epoch:
            return 0.0
        
        if epoch >= self.end_epoch:
            return self.amount
        
        progress = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
        return self.amount * progress

    def step(self, epoch: int):
        """
        Perform pruning step.

        Args:
            epoch: Current epoch
        """
        if epoch % self.frequency != 0:
            return
        
        current_amount = self._get_current_amount(epoch)
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Compute threshold
                threshold = self._compute_threshold(module.weight, current_amount)
                
                # Update mask
                self.masks[name] = (torch.abs(module.weight) > threshold).float()
                
                # Apply mask
                module.weight.data *= self.masks[name]

    def apply_masks(self):
        """Apply pruning masks."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                module.weight.data *= self.masks[name]


class FisherPruner:
    """Fisher information-based pruning."""

    def __init__(
        self,
        model: nn.Module,
        amount: float,
        n_samples: int = 1000,
        device: str = "cuda"
    ):
        """
        Initialize Fisher pruner.

        Args:
            model: Model to prune
            amount: Target pruning amount (0-1)
            n_samples: Number of samples for Fisher estimation
            device: Device to use
        """
        self.model = model
        self.amount = amount
        self.n_samples = n_samples
        self.device = device
        
        self.fisher_info = defaultdict(list)
        self.masks = {}
        self._initialize_masks()

    def _initialize_masks(self):
        """Initialize pruning masks."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                self.masks[name] = torch.ones_like(module.weight)

    def _compute_fisher_info(
        self,
        dataloader: torch.utils.data.DataLoader
    ):
        """
        Compute Fisher information.

        Args:
            dataloader: DataLoader for Fisher estimation
        """
        self.model.train()
        n_samples = 0
        
        for batch in dataloader:
            if n_samples >= self.n_samples:
                break
            
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(batch["input_ids"], batch["attention_mask"])
            loss = nn.CrossEntropyLoss()(
                outputs.view(-1, outputs.size(-1)),
                batch["target_ids"].view(-1)
            )
            
            # Backward pass
            loss.backward()
            
            # Accumulate gradients
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    if module.weight.grad is not None:
                        self.fisher_info[name].append(
                            module.weight.grad.data ** 2
                        )
            
            n_samples += batch["input_ids"].size(0)
        
        # Average Fisher information
        for name in self.fisher_info:
            self.fisher_info[name] = torch.stack(self.fisher_info[name]).mean(0)

    def _compute_threshold(
        self,
        fisher: torch.Tensor,
        amount: float
    ) -> float:
        """
        Compute pruning threshold.

        Args:
            fisher: Fisher information tensor
            amount: Pruning amount

        Returns:
            Pruning threshold
        """
        return torch.quantile(fisher, amount)

    def prune(
        self,
        dataloader: torch.utils.data.DataLoader
    ):
        """
        Perform pruning.

        Args:
            dataloader: DataLoader for Fisher estimation
        """
        # Compute Fisher information
        self._compute_fisher_info(dataloader)
        
        # Compute and apply masks
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Compute threshold
                threshold = self._compute_threshold(
                    self.fisher_info[name],
                    self.amount
                )
                
                # Update mask
                self.masks[name] = (self.fisher_info[name] > threshold).float()
                
                # Apply mask
                module.weight.data *= self.masks[name]

    def apply_masks(self):
        """Apply pruning masks."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                module.weight.data *= self.masks[name] 