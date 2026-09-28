from utils import token_f1, normalize_text


def test_token_f1_identity():
    assert token_f1("abc def", "abc def") == 1.0


def test_normalize():
    assert normalize_text("  Better   / Positive ") == "better / positive"
