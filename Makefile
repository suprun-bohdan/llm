.PHONY: help install install-dev install-opt clean test lint format check docs build publish

help:
	@echo "Доступні команди:"
	@echo "  make install        - встановити базові залежності"
	@echo "  make install-dev    - встановити залежності для розробки"
	@echo "  make install-opt    - встановити опціональні залежності"
	@echo "  make clean          - видалити тимчасові файли"
	@echo "  make test           - запустити тести"
	@echo "  make lint           - перевірити код"
	@echo "  make format         - відформатувати код"
	@echo "  make check          - перевірити формат та стиль"
	@echo "  make docs           - згенерувати документацію"
	@echo "  make build          - зібрати пакет"
	@echo "  make publish        - опублікувати пакет"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-opt:
	pip install -e ".[opt]"

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	find . -type d -name ".tox" -exec rm -rf {} +
	find . -type d -name ".nox" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".hypothesis" -exec rm -rf {} +
	find . -type d -name "logs" -exec rm -rf {} +
	find . -type d -name "tensorboard" -exec rm -rf {} +
	find . -type d -name "wandb" -exec rm -rf {} +
	find . -type d -name "checkpoints" -exec rm -rf {} +
	find . -type d -name "output" -exec rm -rf {} +
	find . -type d -name "data" -exec rm -rf {} +
	find . -type d -name "node_modules" -exec rm -rf {} +
	find . -type d -name ".cache" -exec rm -rf {} +

test:
	pytest tests/ -v --cov=model --cov-report=term-missing

lint:
	flake8 model/ tests/
	mypy model/ tests/
	bandit -r model/ -c pyproject.toml
	ruff check model/ tests/

format:
	black model/ tests/
	isort model/ tests/
	ruff format model/ tests/

check: format lint test

docs:
	sphinx-build -b html docs/ docs/_build/html/

build: clean
	python setup.py sdist bdist_wheel

publish: build
	twine upload dist/* 