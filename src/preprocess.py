"""text preprocessing for LDA: tokenize, lemma, stopwords."""
import re

import spacy
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


def tokenize(text, min_token_len=3, allowed_pos=("NOUN", "ADJ", "VERB", "ADV")):
    """spacy tokenize -> lemma, drop stopwords + short tokens."""
    text = basic_clean(text)
    sw = set(stopwords.words("english"))
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
        out.append(lemma)
    return out
</content>
</invoke>