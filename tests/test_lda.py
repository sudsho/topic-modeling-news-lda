"""smoke test that LDA training returns a usable model."""
import pytest

from src.lda import train_lda
from src.topics import all_labels, top_words


# tiny synthetic "news" so we don't need 20newsgroups for unit tests
DOCS = [
    "the team won the game last night thanks to a great goal".split(),
    "hockey players scored two goals in the third period".split(),
    "the new computer has a faster processor and more memory".split(),
    "this software needs a driver update for the printer".split(),
    "scientists found water on mars during the recent mission".split(),
    "the space probe returned data about the planet surface".split(),
    "voters are debating the new amendment to the law".split(),
    "the senator proposed a bill on tax reform this week".split(),
] * 4  # repeat so dictionary survives filter_extremes


@pytest.mark.slow
def test_train_lda_returns_model():
    lda, dictionary, corpus = train_lda(
        DOCS, num_topics=3, passes=2, iterations=10, random_state=0,
    )
    assert lda.num_topics == 3
    assert len(corpus) == len(DOCS)
    # every topic should have at least 1 word
    for t in range(3):
        assert len(top_words(lda, t, n=5)) >= 1


@pytest.mark.slow
def test_all_labels_length():
    lda, _, _ = train_lda(
        DOCS, num_topics=2, passes=1, iterations=5, random_state=0,
    )
    labels = all_labels(lda, n=2)
    assert len(labels) == 2
    assert all("/" in lab for lab in labels)
</content>
</invoke>