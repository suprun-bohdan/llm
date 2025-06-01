"""
Тести для токенізатора.
"""
import pytest
from tokenizer.simple_tokenizer import SimpleTokenizer


@pytest.fixture
def tokenizer():
    """Фікстура для токенізатора."""
    return SimpleTokenizer(
        vocab_size=100,
        min_freq=2,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"]
    )


def test_tokenizer_init(tokenizer):
    """Тест ініціалізації токенізатора."""
    assert tokenizer.vocab_size == 100
    assert tokenizer.min_freq == 2
    assert tokenizer.special_tokens == ["<pad>", "<unk>", "<bos>", "<eos>"]
    assert not tokenizer.is_trained
    assert len(tokenizer.token_to_id) == 0
    assert len(tokenizer.id_to_token) == 0
    assert len(tokenizer.merges) == 0


def test_preprocess_text(tokenizer):
    """Тест попередньої обробки тексту."""
    text = "Привіт, світ!  Як   справи?"
    processed = tokenizer.preprocess_text(text)
    assert processed == "привіт , світ ! як справи ?"


def test_tokenizer_train(tokenizer):
    """Тест навчання токенізатора."""
    texts = [
        "привіт світ",
        "привіт світ",
        "як справи",
        "як справи",
        "все добре",
        "все добре"
    ]
    
    tokenizer.train(texts)
    
    assert tokenizer.is_trained
    assert len(tokenizer.token_to_id) > 0
    assert len(tokenizer.id_to_token) > 0
    assert len(tokenizer.merges) > 0
    
    # Перевірка спеціальних токенів
    for token in tokenizer.special_tokens:
        assert token in tokenizer.token_to_id
    
    # Перевірка базових символів
    for text in texts:
        for char in text:
            if char not in " ":
                assert char in tokenizer.token_to_id


def test_tokenizer_encode_decode(tokenizer):
    """Тест кодування та декодування."""
    texts = [
        "привіт світ",
        "привіт світ",
        "як справи",
        "як справи",
        "все добре",
        "все добре"
    ]
    
    tokenizer.train(texts)
    
    # Тест кодування
    text = "привіт світ"
    token_ids = tokenizer.encode(text)
    assert isinstance(token_ids, list)
    assert all(isinstance(id_, int) for id_ in token_ids)
    assert len(token_ids) > 0
    
    # Тест декодування
    decoded_text = tokenizer.decode(token_ids)
    assert isinstance(decoded_text, str)
    assert len(decoded_text) > 0
    
    # Тест невідомих символів
    unknown_text = "xyz"
    unknown_ids = tokenizer.encode(unknown_text)
    assert all(id_ == tokenizer.token_to_id["<unk>"] for id_ in unknown_ids)


def test_tokenizer_save_load(tmp_path, tokenizer):
    """Тест збереження та завантаження токенізатора."""
    texts = [
        "привіт світ",
        "привіт світ",
        "як справи",
        "як справи",
        "все добре",
        "все добре"
    ]
    
    tokenizer.train(texts)
    
    # Збереження
    save_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(save_path))
    assert save_path.exists()
    
    # Завантаження
    loaded_tokenizer = SimpleTokenizer.load(str(save_path))
    
    # Перевірка параметрів
    assert loaded_tokenizer.vocab_size == tokenizer.vocab_size
    assert loaded_tokenizer.min_freq == tokenizer.min_freq
    assert loaded_tokenizer.special_tokens == tokenizer.special_tokens
    assert loaded_tokenizer.is_trained
    
    # Перевірка словників
    assert loaded_tokenizer.token_to_id == tokenizer.token_to_id
    assert loaded_tokenizer.id_to_token == tokenizer.id_to_token
    assert loaded_tokenizer.merges == tokenizer.merges
    
    # Перевірка функціональності
    text = "привіт світ"
    assert loaded_tokenizer.encode(text) == tokenizer.encode(text)
    assert loaded_tokenizer.decode(tokenizer.encode(text)) == tokenizer.decode(tokenizer.encode(text))


def test_tokenizer_untrained_errors(tokenizer):
    """Тест помилок при використанні ненавченого токенізатора."""
    with pytest.raises(RuntimeError):
        tokenizer.encode("тест")
    
    with pytest.raises(RuntimeError):
        tokenizer.decode([1, 2, 3])
    
    with pytest.raises(RuntimeError):
        tokenizer.save("test.json") 