"""text preprocessing for LDA: tokenize, lemma, stopwords."""
import re

import spacy
from gensim.models.phrases import Phrases, Phraser
from nltk.corpus import stopwords


_NLP = None


def get_nlp():
    """lazy spacy load. disable parser/ner since we only need lemma+pos."""
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    return _NLP


def basic_clean(text):
    """light regex cleanup: emails, urls, multiple whitespace, digits."""
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


EXTRA_STOPWORDS = {
    # newsgroup boilerplate that survives header stripping
    "say", "would", "could", "get", "go", "know", "think", "make", "want",
    "thing", "people", "way", "lot", "use", "well", "much", "even", "also",
    "really", "actually", "maybe", "sure", "subject", "line", "writes",
    "edu", "com", "org", "article", "post", "newsgroup",
}


def tokenize(text, min_token_len=3, allowed_pos=("NOUN", "ADJ", "VERB", "ADV"),
             extra_stopwords=None):
    """spacy tokenize -> lemma, drop stopwords + short tokens."""
    text = basic_clean(text)
    sw = set(stopwords.words("english"))
    sw.update(EXTRA_STOPWORDS)
    if extra_stopwords:
        sw.update(extra_stopwords)
    doc = get_nlp()(text)
    out = []
    for tok in doc:
        if tok.is_stop or tok.is_punct or tok.is_space:
            continue
        if allowed_pos and tok.pos_ not in allowed_pos:
            continue
        lemma = tok.lemma_.strip().lower()
        if len(lemma) < min_token_len or lemma in sw:
            continue
        if not lemma.isalpha():
            continue
        out.append(lemma)
    return out


def add_bigrams(token_lists, min_count=20, threshold=50):
    """Train a gensim Phrases model on token_lists, return bigrammed lists.

    Note: this is fitted on whatever you pass in. For trigrams, call again
    on the bigrammed output.
    """
    phrases = Phrases(token_lists, min_count=min_count, threshold=threshold)
    bigram = Phraser(phrases)
    return [bigram[toks] for toks in token_lists]
</content>
</invoke>