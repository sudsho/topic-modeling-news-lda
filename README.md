# topic-modeling-news-lda

LDA topic modeling on news data, with an interactive explorer.

## what

Take a news corpus (20 Newsgroups for now), preprocess the text, train an
LDA model with gensim, pick the number of topics `k` by sweeping coherence
(c_v), and look at the topics with pyLDAvis + a small Streamlit app.

## why

A recurring NLP task. Wanted a clean reference setup that goes from raw
posts -> trained model -> interactive viz, without any fancy tricks.

## status

Just bootstrapping. README is a sketch. Code coming.
</content>
</invoke>