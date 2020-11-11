"""pyLDAvis HTML export."""
import argparse
import logging
import os

import pyLDAvis
import pyLDAvis.gensim

from .util import ensure_dir, load_config
from .lda import load


log = logging.getLogger(__name__)


def export_html(lda, corpus, dictionary, out_path):
    ensure_dir(os.path.dirname(out_path) or ".")
    prep = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    pyLDAvis.save_html(prep, out_path)
    return out_path


def main():
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    lda, dictionary = load(cfg["paths"]["model_path"], cfg["paths"]["dictionary_path"])

    # rebuild corpus from saved dict by reading the same data
    from .load_data import load_20newsgroups
    from .preprocess import tokenize, add_bigrams
    from .corpus import to_bow

    docs, _ = load_20newsgroups(subset=cfg["data"]["subset"], remove=cfg["data"]["remove"])
    toks = [tokenize(d) for d in docs]
    if cfg["preprocess"]["ngrams"]["bigrams"]:
        toks = add_bigrams(
            toks,
            min_count=cfg["preprocess"]["ngrams"]["min_count"],
            threshold=cfg["preprocess"]["ngrams"]["threshold"],
        )
    corpus = to_bow(toks, dictionary)

    out = args.out or cfg["paths"]["vis_html"]
    export_html(lda, corpus, dictionary, out)
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
</content>
</invoke>