"""
Тести для скрипту пошуку гіперпараметрів.
"""
import os
import json
import yaml
import pytest
import torch
from unittest.mock import patch, MagicMock
from search_hyperparameters import (
    parse_args,
    load_config,
    setup_logging,
    create_model_fn,
    main
)


@pytest.fixture
def config_path(tmp_path):
    """Фікстура для шляху до конфігурації."""
    config = {
        "model": {
            "d_model": 512,
            "n_heads": 8,
            "n_layers": 6,
            "d_ff": 2048,
            "dropout": 0.1
        },
        "tokenizer": {
            "vocab_size": 32000,
            "min_freq": 2
        },
        "data": {
            "max_length": 512,
            "val_size": 0.1
        },
        "training": {
            "batch_size": 32,
            "num_workers": 4,
            "gradient_clip": 1.0
        },
        "hyperparameter_search": {
            "metric": "val_loss",
            "direction": "minimize",
            "study_name": "test_study",
            "param_space": {
                "d_model": {
                    "type": "int",
                    "min": 256,
                    "max": 1024,
                    "step": 128
                },
                "dropout": {
                    "type": "float",
                    "min": 0.1,
                    "max": 0.5
                }
            },
            "param_grid": {
                "d_model": [256, 512, 1024],
                "dropout": [0.1, 0.2, 0.3]
            }
        }
    }
    
    path = tmp_path / "config.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    
    return path


@pytest.fixture
def data_path(tmp_path):
    """Фікстура для шляху до даних."""
    texts = [
        "Це тестовий текст для перевірки.",
        "Ще один тестовий текст.",
        "І ще один для прикладу."
    ]
    
    path = tmp_path / "data.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(texts))
    
    return path


@pytest.fixture
def output_dir(tmp_path):
    """Фікстура для директорії виводу."""
    return tmp_path / "output"


def test_parse_args():
    """Тест парсингу аргументів."""
    with patch(
        "sys.argv",
        [
            "search_hyperparameters.py",
            "--config", "config.yaml",
            "--data", "data.txt",
            "--output_dir", "output",
            "--search_type", "optuna",
            "--n_trials", "10",
            "--timeout", "3600",
            "--seed", "42"
        ]
    ):
        args = parse_args()
        
        assert args.config == "config.yaml"
        assert args.data == "data.txt"
        assert args.output_dir == "output"
        assert args.search_type == "optuna"
        assert args.n_trials == 10
        assert args.timeout == 3600
        assert args.seed == 42


def test_load_config(config_path):
    """Тест завантаження конфігурації."""
    config = load_config(config_path)
    
    assert isinstance(config, dict)
    assert "model" in config
    assert "tokenizer" in config
    assert "data" in config
    assert "training" in config
    assert "hyperparameter_search" in config
    
    assert config["model"]["d_model"] == 512
    assert config["model"]["n_heads"] == 8
    assert config["model"]["n_layers"] == 6
    assert config["model"]["d_ff"] == 2048
    assert config["model"]["dropout"] == 0.1


def test_setup_logging(output_dir):
    """Тест налаштування логування."""
    logger = setup_logging(output_dir)
    
    assert isinstance(logger, logging.Logger)
    assert logger.level == logging.INFO
    
    log_file = output_dir / "search.log"
    assert log_file.exists()


def test_create_model_fn(config_path):
    """Тест створення функції для моделі."""
    config = load_config(config_path)
    model_fn = create_model_fn(config)
    
    model = model_fn()
    assert isinstance(model, torch.nn.Module)
    assert model.d_model == 512
    assert model.n_heads == 8
    assert model.n_layers == 6
    assert model.d_ff == 2048
    assert model.dropout == 0.1
    
    model = model_fn(d_model=256, dropout=0.2)
    assert model.d_model == 256
    assert model.dropout == 0.2


@patch("search_hyperparameters.HyperparameterSearch")
@patch("search_hyperparameters.GridSearch")
@patch("search_hyperparameters.Tokenizer")
@patch("search_hyperparameters.load_data")
@patch("search_hyperparameters.create_dataset")
@patch("search_hyperparameters.create_dataloaders")
def test_main_optuna(
    mock_create_dataloaders,
    mock_create_dataset,
    mock_load_data,
    mock_tokenizer,
    mock_grid_search,
    mock_hyperparameter_search,
    config_path,
    data_path,
    output_dir
):
    """Тест головної функції з optuna."""
    mock_load_data.return_value = ["текст1", "текст2", "текст3"]
    mock_tokenizer.return_value = MagicMock()
    mock_create_dataset.return_value = (MagicMock(), MagicMock())
    mock_create_dataloaders.return_value = (MagicMock(), MagicMock())
    
    mock_search = MagicMock()
    mock_search.run.return_value = {
        "best_params": {"d_model": 512, "dropout": 0.1},
        "best_value": 0.5
    }
    mock_hyperparameter_search.return_value = mock_search
    
    with patch(
        "sys.argv",
        [
            "search_hyperparameters.py",
            "--config", str(config_path),
            "--data", str(data_path),
            "--output_dir", str(output_dir),
            "--search_type", "optuna",
            "--n_trials", "2"
        ]
    ):
        main()
    
    mock_load_data.assert_called_once_with(str(data_path))
    mock_tokenizer.assert_called_once()
    mock_create_dataset.assert_called_once()
    mock_create_dataloaders.assert_called_once()
    mock_hyperparameter_search.assert_called_once()
    mock_grid_search.assert_not_called()
    mock_search.run.assert_called_once()
    
    results_path = output_dir / "search_results.json"
    assert results_path.exists()
    
    best_config_path = output_dir / "best_config.yaml"
    assert best_config_path.exists()


@patch("search_hyperparameters.HyperparameterSearch")
@patch("search_hyperparameters.GridSearch")
@patch("search_hyperparameters.Tokenizer")
@patch("search_hyperparameters.load_data")
@patch("search_hyperparameters.create_dataset")
@patch("search_hyperparameters.create_dataloaders")
def test_main_grid(
    mock_create_dataloaders,
    mock_create_dataset,
    mock_load_data,
    mock_tokenizer,
    mock_grid_search,
    mock_hyperparameter_search,
    config_path,
    data_path,
    output_dir
):
    """Тест головної функції з grid search."""
    mock_load_data.return_value = ["текст1", "текст2", "текст3"]
    mock_tokenizer.return_value = MagicMock()
    mock_create_dataset.return_value = (MagicMock(), MagicMock())
    mock_create_dataloaders.return_value = (MagicMock(), MagicMock())
    
    mock_search = MagicMock()
    mock_search.run.return_value = {
        "best_params": {"d_model": 512, "dropout": 0.1},
        "best_value": 0.5
    }
    mock_grid_search.return_value = mock_search
    
    with patch(
        "sys.argv",
        [
            "search_hyperparameters.py",
            "--config", str(config_path),
            "--data", str(data_path),
            "--output_dir", str(output_dir),
            "--search_type", "grid"
        ]
    ):
        main()
    
    mock_load_data.assert_called_once_with(str(data_path))
    mock_tokenizer.assert_called_once()
    mock_create_dataset.assert_called_once()
    mock_create_dataloaders.assert_called_once()
    mock_grid_search.assert_called_once()
    mock_hyperparameter_search.assert_not_called()
    mock_search.run.assert_called_once()
    
    results_path = output_dir / "search_results.json"
    assert results_path.exists()
    
    best_config_path = output_dir / "best_config.yaml"
    assert best_config_path.exists()


def test_main_error_handling(config_path, data_path, output_dir):
    """Тест обробки помилок."""
    with patch(
        "sys.argv",
        [
            "search_hyperparameters.py",
            "--config", "неіснуючий.yaml",
            "--data", str(data_path),
            "--output_dir", str(output_dir)
        ]
    ), pytest.raises(SystemExit):
        main()
    
    with patch(
        "sys.argv",
        [
            "search_hyperparameters.py",
            "--config", str(config_path),
            "--data", str(data_path),
            "--output_dir", str(output_dir),
            "--search_type", "некоректний"
        ]
    ), pytest.raises(SystemExit):
        main() 