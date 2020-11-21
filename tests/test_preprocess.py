"""tests for src.preprocess."""
import pytest

from src.preprocess import basic_clean


def test_basic_clean_lowercases():
    assert basic_clean("HELLO World") == "hello world"


def test_basic_clean_strips_emails():
    out = basic_clean("contact me at foo@bar.com please")
    assert "@" not in out
    assert "foo" not in out


def test_basic_clean_strips_urls():
    out = basic_clean("see http://example.com/page for more")
    assert "http" not in out
    assert "example.com" not in out


def test_basic_clean_strips_digits():
    out = basic_clean("year 1996 was great")
    assert "1996" not in out


def test_basic_clean_collapses_whitespace():
    out = basic_clean("a    b\n\tc")
    assert out == "a b c"


@pytest.mark.spacy
def test_tokenize_returns_lemmas():
    # this requires en_core_web_sm to be downloaded; mark so CI can skip
    from src.preprocess import tokenize
    out = tokenize("The cats were running quickly across the rooftops.")
    assert any(t in out for t in ("cat", "run", "quickly", "rooftop"))
</content>
</invoke>