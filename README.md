# topic-modeling-news-lda

LDA topic modeling on the **20 Newsgroups** corpus, with a coherence-driven
choice of `k`, an interactive **pyLDAvis** page, and a small **Streamlit** app
for browsing topics and trying the model on new text.

## the problem

You have a pile of news posts (or any other reasonably long-form text) and
you want a quick, interpretable summary of what they're about. LDA gives you
a fixed number of topics and, for each one, a list of the words that
characterize it, plus a per-document distribution over those topics.

This repo wraps the standard gensim recipe end to end:

1. Pull the **20 Newsgroups** dataset via `sklearn.datasets`.
2. Clean + tokenize with **spaCy** (`en_core_web_sm`), drop stopwords,
   lemmatize, keep nouns/verbs/adjectives/adverbs.
3. Build bigrams with `gensim.models.phrases.Phrases`.
4. Build a `gensim.corpora.Dictionary` and bag-of-words corpus, with
   `filter_extremes` to drop very rare and very common terms.
5. Sweep the **c_v coherence score** over `k = 5..25` to pick a number of
   topics, then retrain at the best `k`.
6. Export an interactive **pyLDAvis** HTML page.
7. Serve a **Streamlit** dashboard with a topic selector, top-N words bar
   chart, coherence-vs-k plot, and a document explorer that runs `predict`
   on either pre-saved sample posts or arbitrary pasted text.

## structure

```
.
|- src/
|  |- load_data.py     20 Newsgroups loader (sklearn)
|  |- preprocess.py    regex cleanup, spaCy tokenize/lemma, bigrams
|  |- corpus.py        Dictionary + BoW
|  |- lda.py           LdaModel / LdaMulticore trainer + save/load
|  |- coherence.py     c_v coherence sweep
|  |- topics.py        helpers for top words and short labels
|  |- predict.py       assign topics to a new document
|  |- visualize.py     pyLDAvis HTML export
|  |- cache.py         tiny pickle cache for tokens
|  |- train.py         end-to-end pipeline (load -> tokens -> sweep -> train -> save)
|  '- util.py          yaml config loader
|- streamlit_app.py    topic explorer (topic selector, bar chart, doc explorer)
|- configs/default.yaml
|- notebooks/eda.ipynb
|- tests/
|- artifacts/          (gitignored; coherence.json, lda.model, lda_vis.html)
|- data/samples/posts.json  small set of demo posts
|- Dockerfile
|- requirements.txt
|- setup.sh            venv + pip install + spacy/nltk downloads
'- Makefile            install / train / vis / app / test
```

## quickstart

```bash
bash setup.sh               # creates .venv, installs deps, downloads spacy/nltk
source .venv/bin/activate
make train                  # full pipeline; writes artifacts/{lda.model, dict.gensim, coherence.json}
make vis                    # writes artifacts/lda_vis.html (open in any browser)
make app                    # starts Streamlit on http://localhost:8501
make test                   # pytest, skips slow markers by default
```

## configuration

Everything is driven from `configs/default.yaml`:

- `data.subset`: `train` / `test` / `all`
- `preprocess.ngrams.bigrams`: turn bigram detection on/off
- `coherence.{k_min,k_max,k_step,metric}`: range of `k` to sweep, and which
  metric (default `c_v`)
- `lda.{passes,iterations,alpha,eta,workers,random_state}`

To try a different range of topics, override the field and rerun
`make train`. The cached tokenized corpus is keyed off the data + preprocess
sections, so changing only `lda` settings doesn't re-tokenize.

## results (on `20news, train, default config`)

The c_v sweep typically peaks around `k=15` (varies a couple of points run
to run). Sample top words for a few of the topics (random run, IDs are not
stable across retrains):

- `team / game / play / season / win / score` -> sports
- `space / mission / launch / orbit / nasa / earth` -> sci.space
- `god / christian / jesus / faith / church / bible` -> talk.religion
- `drive / disk / mac / scsi / system / boot` -> comp.sys.mac.hardware

Run `make vis` and open `artifacts/lda_vis.html` for the full pyLDAvis view.

### screenshots

`artifacts/screenshots/` holds a couple of png exports from the Streamlit app
(topic-words bar chart, coherence-vs-k curve). They are checked in so the
README renders something on a fresh clone:

![topic words](artifacts/screenshots/topic_words.png)

![coherence vs k](artifacts/screenshots/coherence.png)

## deployment

The Dockerfile builds a slim image that runs the Streamlit app. For a
quick local run:

```bash
docker build -t topic-lda .
docker run -p 8501:8501 topic-lda
```

For Heroku container deploy:

```bash
heroku stack:set container -a <app-name>
git push heroku main
```

`heroku.yml` and `Procfile` are both checked in; the `Dockerfile` reads the
`$PORT` env var that Heroku sets at runtime.

## tests

```bash
pytest -q                     # fast unit tests
pytest -q -m "slow"           # also runs the LDA smoke tests
pytest -q -m "slow or spacy"  # plus the spaCy-dependent ones
```
</content>
</invoke>