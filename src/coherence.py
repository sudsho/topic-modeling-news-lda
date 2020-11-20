"""coherence sweep helper.

Trains LDA at several values of k and returns coherence scores so we can
pick a sensible number of topics.
"""
import logging

from gensim.models import CoherenceModel

from .lda import train_lda


log = logging.getLogger(__name__)


def coherence_sweep(token_lists, k_min=5, k_max=25, k_step=5,
                    metric="c_v", passes=5, iterations=50, random_state=42,
                    workers=None):
    """run train+score over a range of k. returns list of (k, score).

    workers > 1 uses LdaMulticore which is faster but uses symmetric alpha.
    Leave at None for single-core LdaModel (lets us keep alpha='auto').
    """
    out = []
    for k in range(k_min, k_max + 1, k_step):
        log.info("training k=%d", k)
        lda, dictionary, _ = train_lda(
            token_lists,
            num_topics=k,
            passes=passes,
            iterations=iterations,
            workers=workers,
            random_state=random_state,
        )
        cm = CoherenceModel(
            model=lda,
            texts=token_lists,
            dictionary=dictionary,
            coherence=metric,
        )
        score = cm.get_coherence()
        log.info("k=%d coherence=%.4f", k, score)
        out.append((k, score))
    return out
</content>
</invoke>