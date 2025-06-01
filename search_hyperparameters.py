"""
Скрипт для пошуку гіперпараметрів.
"""
import os
import argparse
import logging
import yaml
import torch
from torch.utils.data import DataLoader
from model.hyperparameter_search import HyperparameterSearch, GridSearch
from train_from_scratch import Trainer
from model.tokenizer import Tokenizer
from utils.data import load_data, create_dataset, create_dataloaders


def parse_args():
    """Парсинг аргументів командного рядка."""
    parser = argparse.ArgumentParser(description="Пошук гіперпараметрів")
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Шлях до конфігураційного файлу"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Шлях до файлу з даними"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Директорія для збереження результатів"
    )
    parser.add_argument(
        "--search_type",
        type=str,
        choices=["optuna", "grid"],
        default="optuna",
        help="Тип пошуку (optuna або grid)"
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=100,
        help="Кількість спроб для optuna"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Максимальний час пошуку в секундах"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Зерно для відтворюваності"
    )
    
    return parser.parse_args()


def load_config(path):
    """Завантаження конфігурації."""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(output_dir):
    """Налаштування логування."""
    os.makedirs(output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "search.log")),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def create_model_fn(config):
    """Створення функції для створення моделі."""
    def model_fn(**params):
        model_config = config["model"].copy()
        model_config.update(params)
        
        from model.model import TransformerModel
        return TransformerModel(**model_config)
    
    return model_fn


def main():
    """Головна функція."""
    args = parse_args()
    
    logger = setup_logging(args.output_dir)
    logger.info("Початок пошуку гіперпараметрів")
    
    config = load_config(args.config)
    logger.info(f"Завантажено конфігурацію з {args.config}")
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    texts = load_data(args.data)
    logger.info(f"Завантажено {len(texts)} текстів")
    
    tokenizer = Tokenizer(
        vocab_size=config["tokenizer"]["vocab_size"],
        min_freq=config["tokenizer"]["min_freq"]
    )
    tokenizer.train(texts)
    logger.info("Навчено токенізатор")
    
    train_dataset, val_dataset = create_dataset(
        texts,
        tokenizer,
        max_length=config["data"]["max_length"],
        val_size=config["data"]["val_size"]
    )
    
    train_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"]
    )
    logger.info("Створено завантажувачі даних")
    
    model_fn = create_model_fn(config)
    
    if args.search_type == "optuna":
        search = HyperparameterSearch(
            model_fn=model_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            param_space=config["hyperparameter_search"]["param_space"],
            metric=config["hyperparameter_search"]["metric"],
            direction=config["hyperparameter_search"]["direction"],
            n_trials=args.n_trials,
            timeout=args.timeout,
            study_name=config["hyperparameter_search"].get("study_name"),
            storage=config["hyperparameter_search"].get("storage"),
            load_if_exists=config["hyperparameter_search"].get("load_if_exists", False)
        )
        logger.info("Створено пошук з optuna")
    else:
        search = GridSearch(
            model_fn=model_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            param_grid=config["hyperparameter_search"]["param_grid"],
            metric=config["hyperparameter_search"]["metric"],
            direction=config["hyperparameter_search"]["direction"]
        )
        logger.info("Створено пошук по сітці")
    
    logger.info("Початок пошуку")
    results = search.run()
    
    results_path = os.path.join(args.output_dir, "search_results.json")
    search.save_results(results_path)
    logger.info(f"Результати збережено у {results_path}")
    
    logger.info("Найкращі параметри:")
    for name, value in results["best_params"].items():
        logger.info(f"  {name}: {value}")
    
    logger.info(f"Найкраще значення метрики: {results['best_value']}")
    
    best_config = config.copy()
    best_config["model"].update(results["best_params"])
    
    best_config_path = os.path.join(args.output_dir, "best_config.yaml")
    with open(best_config_path, "w", encoding="utf-8") as f:
        yaml.dump(best_config, f, default_flow_style=False)
    logger.info(f"Найкращу конфігурацію збережено у {best_config_path}")


if __name__ == "__main__":
    main() 