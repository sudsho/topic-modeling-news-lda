"""tests for src.predict - uses a tiny LDA model trained inside the test."""
import pytest

from src.lda import train_lda
from src.predict import predict_topics


SPORTS = "the team scored a goal in the third period of the hockey game".split()
SPACE = "the rocket carried the satellite into orbit around the moon".split()
POLITICS = "the senator passed a new bill on tax reform yesterday".split()

DOCS = [SPORTS, SPACE, POLITICS] * 8


@pytest.mark.slow
def test_predict_topics_returns_sorted():
    lda, dictionary, _ = train_lda(
        DOCS, num_topics=3, passes=2, iterations=10, random_state=0,
    )
    out = predict_topics("hockey goal team game", lda, dictionary, min_prob=0.0)
    assert out, "expected at least one topic above threshold 0"
    # sorted by descending prob
    probs = [p for _, p in out]
    assert probs == sorted(probs, reverse=True)


@pytest.mark.slow
def test_predict_threshold_filters():
    lda, dictionary, _ = train_lda(
        DOCS, num_topics=3, passes=2, iterations=10, random_state=0,
    )
    high = predict_topics("hockey goal team", lda, dictionary, min_prob=0.99)
    # no topic will exceed 99% on a 3-topic toy model, so empty
    assert high == [] or all(p >= 0.99 for _, p in high)
</content>
</invoke>