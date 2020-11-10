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
</content>
</invoke>