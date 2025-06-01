"""
Модуль для логування.
"""
import os
import time
from typing import Optional, Dict, Any
from torch.utils.tensorboard import SummaryWriter
import torch


class Logger:
    """Логер для навчання."""

    def __init__(
        self,
        log_dir: str = "runs",
        experiment_name: Optional[str] = None
    ):
        """
        Ініціалізація.

        Args:
            log_dir: Директорія для логів
            experiment_name: Назва експерименту
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
        Логування метрик.

        Args:
            metrics: Словник метрик
            step: Крок
            prefix: Префікс для назв метрик
        """
        for name, value in metrics.items():
            if prefix:
                name = f"{prefix}/{name}"
            self.writer.add_scalar(name, value, step)

    def log_config(self, config: Dict[str, Any]) -> None:
        """
        Логування конфігурації.

        Args:
            config: Конфігурація
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
        Логування графу моделі.

        Args:
            model: Модель
            input_shape: Форма входу
        """
        try:
            dummy_input = torch.zeros(input_shape, dtype=torch.long)
            self.writer.add_graph(model, dummy_input)
        except Exception as e:
            print(f"Помилка логування графу моделі: {e}")

    def close(self) -> None:
        """Закриття логера."""
        self.writer.close() 