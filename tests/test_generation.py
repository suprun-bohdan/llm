"""
Тести для генерації тексту.
"""
import pytest
import torch
from model.transformer import TransformerModel
from tokenizer.simple_tokenizer import SimpleTokenizer


@pytest.fixture
def model_and_tokenizer():
    """Фікстура для моделі та токенізатора."""
    # Параметри
    vocab_size = 1000
    d_model = 64
    n_heads = 4
    n_layers = 2
    d_ff = 256
    max_seq_len = 32
    
    # Створення моделі
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len
    )
    
    # Створення токенізатора
    tokenizer = SimpleTokenizer(
        vocab_size=vocab_size,
        min_freq=2,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"]
    )
    
    # Навчання токенізатора на простому корпусі
    texts = [
        "привіт світ",
        "привіт світ",
        "як справи",
        "як справи",
        "все добре",
        "все добре"
    ]
    tokenizer.train(texts)
    
    return model, tokenizer


def test_greedy_sampling(model_and_tokenizer):
    """Тест жадібної вибірки."""
    model, tokenizer = model_and_tokenizer
    model.eval()
    
    # Підготовка вхідних даних
    input_text = "привіт"
    input_ids = torch.tensor([tokenizer.encode(input_text)])
    
    # Генерація
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_length=10,
            strategy="greedy",
            temperature=0.0,  # Жадібна вибірка
            pad_token_id=tokenizer.token_to_id["<pad>"],
            eos_token_id=tokenizer.token_to_id["<eos>"]
        )
    
    # Перевірки
    assert output_ids.shape[0] == 1  # batch_size
    assert output_ids.shape[1] == 10  # max_length
    assert output_ids[0, 0] == input_ids[0, 0]  # Перший токен зберігається
    assert all(id_ != tokenizer.token_to_id["<pad>"] for id_ in output_ids[0, :len(input_ids[0])])


def test_temperature_sampling(model_and_tokenizer):
    """Тест вибірки з температурою."""
    model, tokenizer = model_and_tokenizer
    model.eval()
    
    # Підготовка вхідних даних
    input_text = "привіт"
    input_ids = torch.tensor([tokenizer.encode(input_text)])
    
    # Генерація з різними температурами
    temperatures = [0.5, 1.0, 2.0]
    outputs = []
    
    with torch.no_grad():
        for temp in temperatures:
            output_ids = model.generate(
                input_ids=input_ids,
                max_length=10,
                strategy="greedy",
                temperature=temp,
                pad_token_id=tokenizer.token_to_id["<pad>"],
                eos_token_id=tokenizer.token_to_id["<eos>"]
            )
            outputs.append(output_ids)
    
    # Перевірки
    for output_ids in outputs:
        assert output_ids.shape[0] == 1
        assert output_ids.shape[1] == 10
        assert output_ids[0, 0] == input_ids[0, 0]
    
    # При високій температурі послідовності повинні відрізнятися
    if torch.cuda.is_available():
        assert not torch.allclose(outputs[0], outputs[2])


def test_top_k_sampling(model_and_tokenizer):
    """Тест top-k вибірки."""
    model, tokenizer = model_and_tokenizer
    model.eval()
    
    # Підготовка вхідних даних
    input_text = "привіт"
    input_ids = torch.tensor([tokenizer.encode(input_text)])
    
    # Генерація з різними k
    k_values = [1, 5, 10]
    outputs = []
    
    with torch.no_grad():
        for k in k_values:
            output_ids = model.generate(
                input_ids=input_ids,
                max_length=10,
                strategy="top_k",
                top_k=k,
                temperature=1.0,
                pad_token_id=tokenizer.token_to_id["<pad>"],
                eos_token_id=tokenizer.token_to_id["<eos>"]
            )
            outputs.append(output_ids)
    
    # Перевірки
    for output_ids in outputs:
        assert output_ids.shape[0] == 1
        assert output_ids.shape[1] == 10
        assert output_ids[0, 0] == input_ids[0, 0]
    
    # При k=1 результат повинен бути таким самим як при жадібній вибірці
    greedy_output = model.generate(
        input_ids=input_ids,
        max_length=10,
        strategy="greedy",
        temperature=0.0,
        pad_token_id=tokenizer.token_to_id["<pad>"],
        eos_token_id=tokenizer.token_to_id["<eos>"]
    )
    assert torch.allclose(outputs[0], greedy_output)


def test_top_p_sampling(model_and_tokenizer):
    """Тест top-p (nucleus) вибірки."""
    model, tokenizer = model_and_tokenizer
    model.eval()
    
    # Підготовка вхідних даних
    input_text = "привіт"
    input_ids = torch.tensor([tokenizer.encode(input_text)])
    
    # Генерація з різними p
    p_values = [0.1, 0.5, 0.9]
    outputs = []
    
    with torch.no_grad():
        for p in p_values:
            output_ids = model.generate(
                input_ids=input_ids,
                max_length=10,
                strategy="top_p",
                top_p=p,
                temperature=1.0,
                pad_token_id=tokenizer.token_to_id["<pad>"],
                eos_token_id=tokenizer.token_to_id["<eos>"]
            )
            outputs.append(output_ids)
    
    # Перевірки
    for output_ids in outputs:
        assert output_ids.shape[0] == 1
        assert output_ids.shape[1] == 10
        assert output_ids[0, 0] == input_ids[0, 0]
    
    # При p=1.0 результат повинен бути таким самим як при звичайній вибірці
    normal_output = model.generate(
        input_ids=input_ids,
        max_length=10,
        strategy="greedy",
        temperature=1.0,
        pad_token_id=tokenizer.token_to_id["<pad>"],
        eos_token_id=tokenizer.token_to_id["<eos>"]
    )
    assert torch.allclose(outputs[2], normal_output)


def test_beam_search(model_and_tokenizer):
    """Тест пошуку по променю."""
    model, tokenizer = model_and_tokenizer
    model.eval()
    
    # Підготовка вхідних даних
    input_text = "привіт"
    input_ids = torch.tensor([tokenizer.encode(input_text)])
    
    # Генерація з різними розмірами променя
    beam_sizes = [1, 2, 4]
    outputs = []
    
    with torch.no_grad():
        for beam_size in beam_sizes:
            output_ids = model.generate(
                input_ids=input_ids,
                max_length=10,
                strategy="beam",
                beam_size=beam_size,
                temperature=1.0,
                length_penalty=1.0,
                pad_token_id=tokenizer.token_to_id["<pad>"],
                eos_token_id=tokenizer.token_to_id["<eos>"]
            )
            outputs.append(output_ids)
    
    # Перевірки
    for output_ids in outputs:
        assert output_ids.shape[0] == 1
        assert output_ids.shape[1] == 10
        assert output_ids[0, 0] == input_ids[0, 0]
    
    # При beam_size=1 результат повинен бути таким самим як при жадібній вибірці
    greedy_output = model.generate(
        input_ids=input_ids,
        max_length=10,
        strategy="greedy",
        temperature=0.0,
        pad_token_id=tokenizer.token_to_id["<pad>"],
        eos_token_id=tokenizer.token_to_id["<eos>"]
    )
    assert torch.allclose(outputs[0], greedy_output)


def test_repetition_penalty(model_and_tokenizer):
    """Тест штрафу за повторення."""
    model, tokenizer = model_and_tokenizer
    model.eval()
    
    # Підготовка вхідних даних
    input_text = "привіт привіт"  # Навмисне повторення
    input_ids = torch.tensor([tokenizer.encode(input_text)])
    
    # Генерація з різними штрафами
    penalties = [1.0, 2.0, 5.0]
    outputs = []
    
    with torch.no_grad():
        for penalty in penalties:
            output_ids = model.generate(
                input_ids=input_ids,
                max_length=10,
                strategy="greedy",
                temperature=1.0,
                repetition_penalty=penalty,
                pad_token_id=tokenizer.token_to_id["<pad>"],
                eos_token_id=tokenizer.token_to_id["<eos>"]
            )
            outputs.append(output_ids)
    
    # Перевірки
    for output_ids in outputs:
        assert output_ids.shape[0] == 1
        assert output_ids.shape[1] == 10
        assert output_ids[0, 0] == input_ids[0, 0]
    
    # При високому штрафі кількість повторень повинна зменшитися
    def count_repetitions(ids):
        ids = ids[0].tolist()  # Беремо першу послідовність
        return sum(1 for i in range(len(ids)-1) if ids[i] == ids[i+1])
    
    reps = [count_repetitions(output) for output in outputs]
    assert reps[0] >= reps[1] >= reps[2]


def test_invalid_strategy(model_and_tokenizer):
    """Тест невалідної стратегії вибірки."""
    model, tokenizer = model_and_tokenizer
    model.eval()
    
    # Підготовка вхідних даних
    input_text = "привіт"
    input_ids = torch.tensor([tokenizer.encode(input_text)])
    
    # Спроба використання невалідної стратегії
    with pytest.raises(ValueError):
        with torch.no_grad():
            model.generate(
                input_ids=input_ids,
                max_length=10,
                strategy="invalid_strategy",
                pad_token_id=tokenizer.token_to_id["<pad>"],
                eos_token_id=tokenizer.token_to_id["<eos>"]
            )


def test_eos_token_handling(model_and_tokenizer):
    """Тест обробки токена кінця послідовності."""
    model, tokenizer = model_and_tokenizer
    model.eval()
    
    # Підготовка вхідних даних з EOS токеном
    input_text = "привіт"
    input_ids = torch.tensor([tokenizer.encode(input_text)])
    input_ids[0, -1] = tokenizer.token_to_id["<eos>"]  # Додаємо EOS токен
    
    # Генерація
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_length=10,
            strategy="greedy",
            temperature=1.0,
            pad_token_id=tokenizer.token_to_id["<pad>"],
            eos_token_id=tokenizer.token_to_id["<eos>"]
        )
    
    # Перевірки
    assert output_ids.shape[0] == 1
    assert output_ids.shape[1] == 10
    assert output_ids[0, 0] == input_ids[0, 0]
    assert output_ids[0, -1] == tokenizer.token_to_id["<pad>"]  # Останній токен - падінгування 