"""build a gensim Dictionary and bag-of-words corpus from token lists."""
from gensim.corpora import Dictionary


def build_dictionary(token_lists, no_below=10, no_above=0.5, keep_n=100000):
    """create a Dictionary and apply gensim's standard filter_extremes."""
    d = Dictionary(token_lists)
    d.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    d.compactify()
    return d


def to_bow(token_lists, dictionary):
    return [dictionary.doc2bow(toks) for toks in token_lists]
</content>
</invoke>