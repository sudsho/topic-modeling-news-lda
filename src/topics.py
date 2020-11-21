"""small helpers for working with trained topics: top words, short labels."""


def top_words(lda, topic_id, n=10):
    return [w for w, _ in lda.show_topic(topic_id, topn=n)]


def short_label(lda, topic_id, n=3, sep="/"):
    """3-word label per topic, useful in dropdowns."""
    return sep.join(top_words(lda, topic_id, n=n))


def all_labels(lda, n=3):
    return [short_label(lda, t, n=n) for t in range(lda.num_topics)]
</content>
</invoke>