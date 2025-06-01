# Базовий образ
FROM python:3.8-slim

# Метадані
LABEL maintainer="LLM from Scratch Team <your.email@example.com>"
LABEL description="Мовна модель з нуля з оптимізаціями"
LABEL version="0.1.0"

# Встановлення системних залежностей
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Створення робочої директорії
WORKDIR /app

# Копіювання файлів проєкту
COPY . .

# Встановлення залежностей
RUN pip install --no-cache-dir -e ".[dev,opt]"

# Налаштування змінних середовища
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=""

# Створення користувача
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Точка входу
ENTRYPOINT ["python"]

# Команда за замовчуванням
CMD ["--help"] 