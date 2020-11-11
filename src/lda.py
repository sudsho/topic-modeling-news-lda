"""train an LDA model with gensim."""
import logging
import os

from gensim.models import LdaModel, LdaMulticore

from .corpus import build_dictionary, to_bow
from .util import ensure_dir


log = logging.getLogger(__name__)


def train_lda(token_lists, num_topics=10, passes=10, iterations=50,
              alpha="auto", eta="auto", workers=None, random_state=42):
    """train and return (lda, dictionary, corpus)."""
    dictionary = build_dictionary(token_lists)
    corpus = to_bow(token_lists, dictionary)

    if workers and workers > 1:
        # LdaMulticore does not support alpha="auto"
        a = "symmetric" if alpha == "auto" else alpha
        e = None if eta == "auto" else eta
        lda = LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=passes,
            iterations=iterations,
            workers=workers,
            alpha=a,
            eta=e,
            random_state=random_state,
        )
    else:
        lda = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=passes,
            iterations=iterations,
            alpha=alpha,
            eta=eta,
            random_state=random_state,
        )

    return lda, dictionary, corpus


def save(lda, dictionary, model_path, dict_path):
    ensure_dir(os.path.dirname(model_path))
    lda.save(model_path)
    dictionary.save(dict_path)


def load(model_path, dict_path):
    from gensim.corpora import Dictionary
    return LdaModel.load(model_path), Dictionary.load(dict_path)


def main():
    import argparse
    from .load_data import load_20newsgroups
    from .preprocess import tokenize, add_bigrams
    from .util import load_config

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()

    cfg = load_config(args.config)

    log.info("loading data")
    docs, _ = load_20newsgroups(
        subset=cfg["data"]["subset"],
        remove=cfg["data"]["remove"],
    )
    log.info("got %d docs", len(docs))

    log.info("preprocessing")
    toks = [tokenize(d, min_token_len=cfg["preprocess"]["min_token_len"]) for d in docs]
    if cfg["preprocess"]["ngrams"]["bigrams"]:
        toks = add_bigrams(
            toks,
            min_count=cfg["preprocess"]["ngrams"]["min_count"],
            threshold=cfg["preprocess"]["ngrams"]["threshold"],
        )

    log.info("training lda")
    lda, dictionary, _ = train_lda(
        toks,
        num_topics=cfg["lda"]["num_topics"],
        passes=cfg["lda"]["passes"],
        iterations=cfg["lda"]["iterations"],
        alpha=cfg["lda"]["alpha"],
        eta=cfg["lda"]["eta"],
        workers=cfg["lda"].get("workers"),
        random_state=cfg["lda"]["random_state"],
    )

    save(lda, dictionary, cfg["paths"]["model_path"], cfg["paths"]["dictionary_path"])
    log.info("saved model -> %s", cfg["paths"]["model_path"])

    for tid, words in lda.show_topics(num_topics=-1, num_words=8, formatted=False):
        top = ", ".join(w for w, _ in words)
        log.info("topic %d: %s", tid, top)


if __name__ == "__main__":
    main()
</content>
</invoke>