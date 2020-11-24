"""tests for the tiny pickle cache."""
import os

from src import cache


def test_cache_key_stable():
    a = cache.cache_key({"x": 1, "y": [2, 3]})
    b = cache.cache_key({"x": 1, "y": [2, 3]})
    assert a == b


def test_cache_key_changes_with_input():
    a = cache.cache_key({"x": 1})
    b = cache.cache_key({"x": 2})
    assert a != b


def test_get_missing_returns_none(tmp_path):
    p = str(tmp_path / "missing.pkl")
    assert cache.get(p) is None


def test_put_and_get_roundtrip(tmp_path):
    p = str(tmp_path / "obj.pkl")
    obj = {"a": [1, 2, 3]}
    cache.put(p, obj)
    assert os.path.exists(p)
    assert cache.get(p) == obj
</content>
</invoke>