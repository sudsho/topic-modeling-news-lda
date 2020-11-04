"""
Loaders for the news corpus.

Currently only 20 Newsgroups via sklearn. AG News might come later.
"""
from sklearn.datasets import fetch_20newsgroups


DEFAULT_REMOVE = ("headers", "footers", "quotes")


def load_20newsgroups(subset="train", remove=DEFAULT_REMOVE, categories=None):
    """Return a list of raw documents and matching category names."""
    bunch = fetch_20newsgroups(
        subset=subset,
        remove=tuple(remove),
        categories=categories,
        shuffle=True,
        random_state=42,
    )
    return list(bunch.data), [bunch.target_names[t] for t in bunch.target]


if __name__ == "__main__":
    docs, labels = load_20newsgroups()
    print("loaded", len(docs), "docs")
    print("first label:", labels[0])
</content>
</invoke>