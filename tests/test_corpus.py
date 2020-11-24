"""tests for src.corpus."""
from src.corpus import build_dictionary, to_bow


def test_build_dictionary_basic():
    docs = [
        ["apple", "banana", "apple"],
        ["banana", "cherry"],
        ["apple", "cherry"],
    ]
    d = build_dictionary(docs, no_below=1, no_above=1.0)
    tokens = set(d.values())
    assert {"apple", "banana", "cherry"} <= tokens


def test_to_bow_returns_pairs():
    docs = [["a", "a", "b"], ["b", "c"]]
    d = build_dictionary(docs, no_below=1, no_above=1.0)
    bow = to_bow(docs, d)
    assert len(bow) == 2
    # first doc has 2 unique words
    assert len(bow[0]) == 2
    counts = {d[i]: c for i, c in bow[0]}
    assert counts == {"a": 2, "b": 1}


def test_filter_extremes_drops_rare():
    docs = [["a", "b"], ["a", "c"], ["a", "d"], ["a", "e"]]
    d = build_dictionary(docs, no_below=2, no_above=1.0)
    # only "a" appears in >=2 docs
    assert "a" in set(d.values())
    assert "b" not in set(d.values())
</content>
</invoke>