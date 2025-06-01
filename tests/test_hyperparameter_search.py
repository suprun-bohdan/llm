"""
Тести для пошуку гіперпараметрів.
"""
import os
import json
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model.hyperparameter_search import HyperparameterSearch, GridSearch


@pytest.fixture
def simple_model_fn():
    """Фікстура для функції створення моделі."""
    def create_model(hidden_size: int = 64, dropout: float = 0.1):
        return nn.Sequential(
            nn.Linear(10, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 5)
        )
    return create_model


@pytest.fixture
def train_loader():
    """Фікстура для тренувального завантажувача."""
    # Створення випадкових даних
    inputs = torch.randn(100, 10)
    labels = torch.randint(0, 5, (100,))
    dataset = TensorDataset(inputs, labels)
    return DataLoader(dataset, batch_size=32, shuffle=True)


@pytest.fixture
def val_loader():
    """Фікстура для валідаційного завантажувача."""
    # Створення випадкових даних
    inputs = torch.randn(20, 10)
    labels = torch.randint(0, 5, (20,))
    dataset = TensorDataset(inputs, labels)
    return DataLoader(dataset, batch_size=32)


@pytest.fixture
def param_space():
    """Фікстура для простору параметрів."""
    return {
        "hidden_size": [32, 64, 128],
        "dropout": [0.1, 0.2, 0.3],
        "learning_rate": {
            "type": "float",
            "min": 1e-4,
            "max": 1e-2,
            "log": True
        },
        "optimizer": {
            "type": "categorical",
            "values": ["adam", "sgd"]
        }
    }


@pytest.fixture
def param_grid():
    """Фікстура для сітки параметрів."""
    return {
        "hidden_size": [32, 64],
        "dropout": [0.1, 0.2],
        "learning_rate": [1e-4, 1e-3],
        "optimizer": ["adam", "sgd"]
    }


def test_hyperparameter_search(
    simple_model_fn,
    train_loader,
    val_loader,
    param_space,
    tmp_path
):
    """Тест пошуку гіперпараметрів."""
    # Створення пошуку
    search = HyperparameterSearch(
        model_fn=simple_model_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        param_space=param_space,
        n_trials=2,  # Мала кількість для тесту
        study_name="test_study",
        storage=f"sqlite:///{tmp_path}/test.db"
    )
    
    # Запуск пошуку
    results = search.run()
    
    # Перевірка результатів
    assert isinstance(results, dict)
    assert "best_params" in results
    assert "best_value" in results
    assert "best_trial" in results
    assert "n_trials" in results
    assert "completed_trials" in results
    assert "pruned_trials" in results
    assert "failed_trials" in results
    assert "all_trials" in results
    
    # Перевірка параметрів
    best_params = results["best_params"]
    assert "hidden_size" in best_params
    assert "dropout" in best_params
    assert "learning_rate" in best_params
    assert "optimizer" in best_params
    
    # Перевірка збереження
    save_path = tmp_path / "results.json"
    search.save_results(str(save_path))
    
    assert save_path.exists()
    with open(save_path, "r", encoding="utf-8") as f:
        saved_results = json.load(f)
    
    assert saved_results == results


def test_grid_search(
    simple_model_fn,
    train_loader,
    val_loader,
    param_grid,
    tmp_path
):
    """Тест пошуку по сітці."""
    # Створення пошуку
    search = GridSearch(
        model_fn=simple_model_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        param_grid=param_grid
    )
    
    # Запуск пошуку
    results = search.run()
    
    # Перевірка результатів
    assert isinstance(results, dict)
    assert "best_params" in results
    assert "best_value" in results
    assert "best_metrics" in results
    assert "all_results" in results
    
    # Перевірка параметрів
    best_params = results["best_params"]
    assert "hidden_size" in best_params
    assert "dropout" in best_params
    assert "learning_rate" in best_params
    assert "optimizer" in best_params
    
    # Перевірка всіх результатів
    all_results = results["all_results"]
    assert len(all_results) == len(list(ParameterGrid(param_grid)))
    
    for result in all_results:
        assert "params" in result
        assert "value" in result or "error" in result
    
    # Перевірка збереження
    save_path = tmp_path / "grid_results.json"
    search.save_results(str(save_path))
    
    assert save_path.exists()
    with open(save_path, "r", encoding="utf-8") as f:
        saved_results = json.load(f)
    
    assert saved_results == results


def test_hyperparameter_search_error_handling(
    simple_model_fn,
    train_loader,
    val_loader,
    tmp_path
):
    """Тест обробки помилок при пошуку гіперпараметрів."""
    # Створення пошуку з некоректними параметрами
    search = HyperparameterSearch(
        model_fn=simple_model_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        param_space={
            "hidden_size": [-1],  # Некоректний розмір
            "dropout": [2.0]  # Некоректна ймовірність
        },
        n_trials=1,
        study_name="test_error_study",
        storage=f"sqlite:///{tmp_path}/test_error.db"
    )
    
    # Запуск пошуку
    results = search.run()
    
    # Перевірка, що всі спроби завершились з помилкою
    assert results["completed_trials"] == 0
    assert results["failed_trials"] > 0


def test_grid_search_error_handling(
    simple_model_fn,
    train_loader,
    val_loader,
    tmp_path
):
    """Тест обробки помилок при пошуку по сітці."""
    # Створення пошуку з некоректними параметрами
    search = GridSearch(
        model_fn=simple_model_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        param_grid={
            "hidden_size": [-1],  # Некоректний розмір
            "dropout": [2.0]  # Некоректна ймовірність
        }
    )
    
    # Запуск пошуку
    results = search.run()
    
    # Перевірка, що всі спроби завершились з помилкою
    assert all("error" in result for result in results["all_results"])
    assert results["best_params"] is None
    assert results["best_value"] == float("inf")  # Для minimize
    assert results["best_metrics"] is None 