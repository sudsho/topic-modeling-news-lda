"""assign topic distribution to a new document."""
import argparse

from .lda import load
from .preprocess import tokenize


def predict_topics(text, lda, dictionary, min_prob=0.05):
    """returns list of (topic_id, prob) sorted desc."""
    toks = tokenize(text)
    bow = dictionary.doc2bow(toks)
    dist = lda.get_document_topics(bow, minimum_probability=min_prob)
    return sorted(dist, key=lambda kv: -kv[1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dict", required=True)
    p.add_argument("--text", required=True)
    args = p.parse_args()

    lda, dictionary = load(args.model, args.dict)
    out = predict_topics(args.text, lda, dictionary)
    for tid, prob in out:
        words = [w for w, _ in lda.show_topic(tid, topn=6)]
        print("topic", tid, "p=%.3f" % prob, "->", ", ".join(words))


if __name__ == "__main__":
    main()
</content>
</invoke>