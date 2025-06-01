"""
Тести для дистиляції моделі.
"""
import pytest
import torch
import torch.nn as nn
from model.distillation import (
    DistillationLoss,
    DistillationTrainer,
    ProgressiveDistillation
)


@pytest.fixture
def teacher_model():
    """Фікстура для моделі-вчителя."""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )


@pytest.fixture
def student_model():
    """Фікстура для моделі-студента."""
    return nn.Sequential(
        nn.Linear(10, 15),
        nn.ReLU(),
        nn.Linear(15, 5)
    )


@pytest.fixture
def optimizer(student_model):
    """Фікстура для оптимізатора."""
    return torch.optim.Adam(student_model.parameters())


@pytest.fixture
def distillation_loss():
    """Фікстура для функції втрат дистиляції."""
    return DistillationLoss(temperature=2.0, alpha=0.5)


@pytest.fixture
def distillation_trainer(teacher_model, student_model, optimizer):
    """Фікстура для тренера дистиляції."""
    return DistillationTrainer(
        student=student_model,
        teacher=teacher_model,
        optimizer=optimizer,
        temperature=2.0,
        alpha=0.5
    )


@pytest.fixture
def progressive_distillation(teacher_model, student_model, optimizer):
    """Фікстура для прогресивної дистиляції."""
    return ProgressiveDistillation(
        student=student_model,
        teacher=teacher_model,
        optimizer=optimizer,
        num_steps=3
    )


def test_distillation_loss(distillation_loss):
    """Тест функції втрат дистиляції."""
    batch_size = 4
    num_classes = 5
    
    # Створення випадкових логітів і міток
    student_logits = torch.randn(batch_size, num_classes)
    teacher_logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Обчислення втрат
    loss = distillation_loss(student_logits, teacher_logits, labels)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Скаляр
    
    # Перевірка різних температур
    loss_t1 = DistillationLoss(temperature=1.0)(
        student_logits,
        teacher_logits,
        labels
    )
    loss_t2 = DistillationLoss(temperature=2.0)(
        student_logits,
        teacher_logits,
        labels
    )
    assert not torch.allclose(loss_t1, loss_t2)
    
    # Перевірка різних ваг
    loss_a1 = DistillationLoss(alpha=0.1)(
        student_logits,
        teacher_logits,
        labels
    )
    loss_a2 = DistillationLoss(alpha=0.9)(
        student_logits,
        teacher_logits,
        labels
    )
    assert not torch.allclose(loss_a1, loss_a2)


def test_distillation_trainer(distillation_trainer):
    """Тест тренера дистиляції."""
    batch_size = 4
    
    # Створення випадкових даних
    inputs = torch.randn(batch_size, 10)
    labels = torch.randint(0, 5, (batch_size,))
    
    # Крок навчання
    metrics = distillation_trainer.train_step(inputs, labels)
    
    assert isinstance(metrics, dict)
    assert "loss" in metrics
    assert "accuracy" in metrics
    assert "kl_div" in metrics
    
    # Перевірка оцінки
    dataloader = [(inputs, labels)]
    eval_metrics = distillation_trainer.evaluate(dataloader)
    
    assert isinstance(eval_metrics, dict)
    assert "loss" in eval_metrics
    assert "accuracy" in eval_metrics
    assert "kl_div" in eval_metrics
    
    # Перевірка збереження/завантаження
    distillation_trainer.save_checkpoint("test_checkpoint.pt")
    
    # Створення нового тренера
    new_trainer = DistillationTrainer(
        student=distillation_trainer.student,
        teacher=distillation_trainer.teacher,
        optimizer=distillation_trainer.optimizer
    )
    
    # Завантаження чекпоінту
    new_trainer.load_checkpoint("test_checkpoint.pt")
    
    # Перевірка, що моделі однакові
    for p1, p2 in zip(
        distillation_trainer.student.parameters(),
        new_trainer.student.parameters()
    ):
        assert torch.allclose(p1, p2)


def test_progressive_distillation(progressive_distillation):
    """Тест прогресивної дистиляції."""
    batch_size = 4
    
    # Створення випадкових даних
    inputs = torch.randn(batch_size, 10)
    labels = torch.randint(0, 5, (batch_size,))
    
    # Перевірка параметрів для різних кроків
    for step in range(progressive_distillation.num_steps):
        temperature, alpha = progressive_distillation.get_step_params(step)
        assert isinstance(temperature, float)
        assert isinstance(alpha, float)
        assert 0 < temperature <= 4.0
        assert 0.5 <= alpha <= 0.9
    
    # Крок навчання
    metrics = progressive_distillation.train_step(inputs, labels, step=0)
    
    assert isinstance(metrics, dict)
    assert "loss" in metrics
    assert "accuracy" in metrics
    assert "kl_div" in metrics
    assert "temperature" in metrics
    assert "alpha" in metrics
    
    # Перевірка навчання епохи
    dataloader = [(inputs, labels)]
    epoch_metrics = progressive_distillation.train_epoch(dataloader, step=0)
    
    assert isinstance(epoch_metrics, dict)
    assert "loss" in epoch_metrics
    assert "accuracy" in epoch_metrics
    assert "kl_div" in epoch_metrics
    assert "temperature" in epoch_metrics
    assert "alpha" in epoch_metrics
    
    # Перевірка оцінки
    eval_metrics = progressive_distillation.evaluate(dataloader, step=0)
    
    assert isinstance(eval_metrics, dict)
    assert "loss" in eval_metrics
    assert "accuracy" in eval_metrics
    assert "kl_div" in eval_metrics
    assert "temperature" in eval_metrics
    assert "alpha" in eval_metrics
    
    # Перевірка збереження/завантаження
    step = 1
    progressive_distillation.save_checkpoint("test_checkpoint.pt", step)
    
    # Створення нового тренера
    new_trainer = ProgressiveDistillation(
        student=progressive_distillation.student,
        teacher=progressive_distillation.teacher,
        optimizer=progressive_distillation.optimizer
    )
    
    # Завантаження чекпоінту
    loaded_step = new_trainer.load_checkpoint("test_checkpoint.pt")
    assert loaded_step == step
    
    # Перевірка, що моделі однакові
    for p1, p2 in zip(
        progressive_distillation.student.parameters(),
        new_trainer.student.parameters()
    ):
        assert torch.allclose(p1, p2) 