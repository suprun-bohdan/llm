"""
Logging module.
"""
import os
import time
from typing import Optional, Dict, Any
from torch.utils.tensorboard import SummaryWriter
import torch
import logging


class Logger:
    """Training logger."""

    def __init__(
        self,
        log_dir: str = "runs",
        experiment_name: Optional[str] = None
    ):
        """
        Initialize.

        Args:
            log_dir: Log directory
            experiment_name: Experiment name
        """
        if experiment_name is None:
            experiment_name = time.strftime("%Y%m%d_%H%M%S")

        self.log_dir = os.path.join(log_dir, experiment_name)
        self.writer = SummaryWriter(self.log_dir)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = ""
    ) -> None:
        """
        Log metrics.

        Args:
            metrics: Dictionary of metrics
            step: Step
            prefix: Metric name prefix
        """
        for name, value in metrics.items():
            if prefix:
                name = f"{prefix}/{name}"
            self.writer.add_scalar(name, value, step)

    def log_config(self, config: Dict[str, Any]) -> None:
        """
        Log configuration.

        Args:
            config: Configuration
        """
        for name, value in config.items():
            if isinstance(value, (int, float, str, bool)):
                self.writer.add_text(f"config/{name}", str(value))

    def log_model_graph(
        self,
        model: Any,
        input_shape: tuple = (1, 256)
    ) -> None:
        """
        Log model graph.

        Args:
            model: Model
            input_shape: Input shape
        """
        try:
            dummy_input = torch.zeros(input_shape, dtype=torch.long)
            self.writer.add_graph(model, dummy_input)
        except Exception as e:
            print(f"Error logging model graph: {e}")

    def close(self) -> None:
        """Close logger."""
        self.writer.close()

def setup_logger(
    name: str,
    log_dir: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Налаштування логера.
    
    Args:
        name: Назва логера
        log_dir: Директорія для логів
        log_file: Назва файлу логу (опціонально)
        level: Рівень логування
        
    Returns:
        Налаштований логер
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if log_file:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, log_file),
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger