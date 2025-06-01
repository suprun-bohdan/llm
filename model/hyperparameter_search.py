"""
Автоматичний пошук гіперпараметрів.
"""
import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import ParameterGrid
from optuna import create_study, Trial
from optuna.trial import TrialState
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from train_from_scratch import Trainer
from model.distillation import DistillationTrainer
from model.quantization import QuantizationAwareTraining


class HyperparameterSearch:
    """Клас для пошуку оптимальних гіперпараметрів."""
    
    def __init__(
        self,
        model_fn,
        train_loader: DataLoader,
        val_loader: DataLoader,
        param_space: Dict[str, Any],
        metric: str = "val_loss",
        direction: str = "minimize",
        n_trials: int = 100,
        timeout: Optional[int] = None,
        pruner: Optional[MedianPruner] = None,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        load_if_exists: bool = False
    ):
        """
        Ініціалізація пошуку гіперпараметрів.
        
        Args:
            model_fn: Функція, що створює модель
            train_loader: Завантажувач тренувальних даних
            val_loader: Завантажувач валідаційних даних
            param_space: Простір гіперпараметрів
            metric: Метрика для оптимізації
            direction: Напрямок оптимізації ("minimize" або "maximize")
            n_trials: Кількість спроб
            timeout: Максимальний час пошуку в секундах
            pruner: Обрізач для припинення неефективних спроб
            study_name: Назва дослідження
            storage: Шлях до зберігання результатів
            load_if_exists: Чи завантажувати існуюче дослідження
        """
        self.model_fn = model_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.param_space = param_space
        self.metric = metric
        self.direction = direction
        self.n_trials = n_trials
        self.timeout = timeout
        
        self.study = create_study(
            study_name=study_name,
            storage=storage,
            sampler=TPESampler(),
            pruner=pruner,
            direction=direction,
            load_if_exists=load_if_exists
        )
        
        self.logger = logging.getLogger(__name__)
    
    def objective(self, trial: Trial) -> float:
        """
        Цільова функція для оптимізації.
        
        Args:
            trial: Спроба оптимізації
            
        Returns:
            Значення метрики
        """
        params = {}
        for name, space in self.param_space.items():
            if isinstance(space, list):
                if all(isinstance(x, (int, float)) for x in space):
                    if all(isinstance(x, int) for x in space):
                        params[name] = trial.suggest_int(name, min(space), max(space))
                    else:
                        params[name] = trial.suggest_float(name, min(space), max(space))
                else:
                    params[name] = trial.suggest_categorical(name, space)
            elif isinstance(space, dict):
                if space["type"] == "int":
                    params[name] = trial.suggest_int(
                        name,
                        space["min"],
                        space["max"],
                        step=space.get("step", 1)
                    )
                elif space["type"] == "float":
                    params[name] = trial.suggest_float(
                        name,
                        space["min"],
                        space["max"],
                        log=space.get("log", False)
                    )
                elif space["type"] == "categorical":
                    params[name] = trial.suggest_categorical(name, space["values"])
        
        model = self.model_fn(**params)
        
        trainer = Trainer(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            **params
        )
        
        try:
            trainer.train(
                epochs=params.get("epochs", 10),
                early_stopping_patience=params.get("early_stopping_patience", 3)
            )
            
            best_metric = trainer.best_metrics[self.metric]
            
            for name, value in trainer.best_metrics.items():
                trial.set_user_attr(name, value)
            
            return best_metric
            
        except Exception as e:
            self.logger.error(f"Помилка при навчанні: {str(e)}")
            raise optuna.TrialPruned()
    
    def run(self) -> Dict[str, Any]:
        """
        Запуск пошуку гіперпараметрів.
        
        Returns:
            Словник з найкращими параметрами
        """
        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout
        )
        
        best_params = self.study.best_params
        best_value = self.study.best_value
        
        results = {
            "best_params": best_params,
            "best_value": best_value,
            "best_trial": self.study.best_trial.number,
            "n_trials": len(self.study.trials),
            "completed_trials": len([t for t in self.study.trials if t.state == TrialState.COMPLETE]),
            "pruned_trials": len([t for t in self.study.trials if t.state == TrialState.PRUNED]),
            "failed_trials": len([t for t in self.study.trials if t.state == TrialState.FAIL]),
            "all_trials": [
                {
                    "number": t.number,
                    "value": t.value,
                    "params": t.params,
                    "state": t.state.name,
                    "user_attrs": t.user_attrs
                }
                for t in self.study.trials
            ]
        }
        
        return results
    
    def save_results(self, path: str) -> None:
        """
        Збереження результатів пошуку.
        
        Args:
            path: Шлях для збереження
        """
        results = self.run()
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Результати збережено у {path}")


class GridSearch:
    """Клас для пошуку по сітці."""
    
    def __init__(
        self,
        model_fn,
        train_loader: DataLoader,
        val_loader: DataLoader,
        param_grid: Dict[str, List[Any]],
        metric: str = "val_loss",
        direction: str = "minimize"
    ):
        """
        Ініціалізація пошуку по сітці.
        
        Args:
            model_fn: Функція, що створює модель
            train_loader: Завантажувач тренувальних даних
            val_loader: Завантажувач валідаційних даних
            param_grid: Сітка параметрів
            metric: Метрика для оптимізації
            direction: Напрямок оптимізації ("minimize" або "maximize")
        """
        self.model_fn = model_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.param_grid = param_grid
        self.metric = metric
        self.direction = direction
        
        self.logger = logging.getLogger(__name__)
    
    def evaluate_params(self, params: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Оцінка параметрів.
        
        Args:
            params: Параметри для оцінки
            
        Returns:
            Кортеж (значення метрики, всі метрики)
        """
        model = self.model_fn(**params)
        
        trainer = Trainer(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            **params
        )
        
        trainer.train(
            epochs=params.get("epochs", 10),
            early_stopping_patience=params.get("early_stopping_patience", 3)
        )
        
        metric_value = trainer.best_metrics[self.metric]
        all_metrics = trainer.best_metrics
        
        return metric_value, all_metrics
    
    def run(self) -> Dict[str, Any]:
        """
        Запуск пошуку по сітці.
        
        Returns:
            Словник з результатами
        """
        results = []
        best_value = float("inf") if self.direction == "minimize" else float("-inf")
        best_params = None
        best_metrics = None
        
        for params in ParameterGrid(self.param_grid):
            try:
                self.logger.info(f"Оцінка параметрів: {params}")
                
                metric_value, all_metrics = self.evaluate_params(params)
                
                results.append({
                    "params": params,
                    "value": metric_value,
                    "metrics": all_metrics
                })
                
                if (
                    (self.direction == "minimize" and metric_value < best_value) or
                    (self.direction == "maximize" and metric_value > best_value)
                ):
                    best_value = metric_value
                    best_params = params
                    best_metrics = all_metrics
                
            except Exception as e:
                self.logger.error(f"Помилка при оцінці параметрів {params}: {str(e)}")
                results.append({
                    "params": params,
                    "error": str(e)
                })
        
        return {
            "best_params": best_params,
            "best_value": best_value,
            "best_metrics": best_metrics,
            "all_results": results
        }
    
    def save_results(self, path: str) -> None:
        """
        Збереження результатів пошуку.
        
        Args:
            path: Шлях для збереження
        """
        results = self.run()
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Результати збережено у {path}") 