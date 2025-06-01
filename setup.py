"""
Налаштування пакету.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("

setup(
    name="llm-from-scratch",
    version="0.1.0",
    author="LLM from Scratch Team",
    author_email="your.email@example.com",
    description="Мовна модель з нуля з оптимізаціями",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/llm-from-scratch",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black",
            "isort",
            "flake8",
            "mypy",
            "pytest",
            "pytest-cov",
            "pytest-benchmark",
            "pre-commit",
        ],
        "opt": [
            "bitsandbytes",
            "accelerate",
            "deepspeed",
            "sentencepiece",
            "protobuf",
        ],
    },
    entry_points={
        "console_scripts": [
            "llm-train=model.train:main",
            "llm-generate=model.generate:main",
            "llm-search=model.search_hyperparameters:main",
        ],
    },
) 