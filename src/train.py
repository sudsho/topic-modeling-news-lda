"""end-to-end training: load -> preprocess -> coherence sweep -> retrain best k."""
import argparse
import json
import logging
import os

from .coherence import coherence_sweep
from .corpus import to_bow
from .lda import save, train_lda
from .load_data import load_20newsgroups
from .preprocess import add_bigrams, tokenize
from .util import ensure_dir, load_config


log = logging.getLogger(__name__)


def run(cfg):
    docs, _ = load_20newsgroups(
        subset=cfg["data"]["subset"],
        remove=cfg["data"]["remove"],
    )
    log.info("got %d docs", len(docs))

    log.info("tokenizing")
    toks = [tokenize(d, min_token_len=cfg["preprocess"]["min_token_len"]) for d in docs]

    if cfg["preprocess"]["ngrams"]["bigrams"]:
        toks = add_bigrams(
            toks,
            min_count=cfg["preprocess"]["ngrams"]["min_count"],
            threshold=cfg["preprocess"]["ngrams"]["threshold"],
        )

    log.info("coherence sweep")
    sweep = coherence_sweep(
        toks,
        k_min=cfg["coherence"]["k_min"],
        k_max=cfg["coherence"]["k_max"],
        k_step=cfg["coherence"]["k_step"],
        metric=cfg["coherence"]["metric"],
        passes=cfg["lda"]["passes"],
        iterations=cfg["lda"]["iterations"],
        random_state=cfg["lda"]["random_state"],
    )
    best_k, best_score = max(sweep, key=lambda kv: kv[1])
    log.info("best k=%d coherence=%.4f", best_k, best_score)

    ensure_dir(cfg["paths"]["artifacts_dir"])
    sweep_out = os.path.join(cfg["paths"]["artifacts_dir"], "coherence.json")
    with open(sweep_out, "w") as f:
        json.dump({"sweep": sweep, "best_k": best_k}, f, indent=2)
    log.info("wrote %s", sweep_out)

    log.info("retrain at best k=%d", best_k)
    lda, dictionary, _ = train_lda(
        toks,
        num_topics=best_k,
        passes=cfg["lda"]["passes"],
        iterations=cfg["lda"]["iterations"],
        alpha=cfg["lda"]["alpha"],
        eta=cfg["lda"]["eta"],
        workers=cfg["lda"].get("workers"),
        random_state=cfg["lda"]["random_state"],
    )
    save(lda, dictionary, cfg["paths"]["model_path"], cfg["paths"]["dictionary_path"])
    log.info("saved -> %s", cfg["paths"]["model_path"])


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
</content>
</invoke>