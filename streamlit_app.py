"""Streamlit topic explorer.

Loads the trained LDA model and lets the user inspect topics, see top words,
and try the model on new text.
"""
import json
import os

import pandas as pd
import streamlit as st

from src.lda import load
from src.predict import predict_topics
from src.util import load_config


st.set_page_config(page_title="20 Newsgroups Topic Explorer", layout="wide")


@st.cache(allow_output_mutation=True)
def get_model(model_path, dict_path):
    return load(model_path, dict_path)


def main():
    cfg = load_config("configs/default.yaml")
    model_path = cfg["paths"]["model_path"]
    dict_path = cfg["paths"]["dictionary_path"]

    if not os.path.exists(model_path):
        st.error("No trained model at %s. Run `make train` first." % model_path)
        return

    lda, dictionary = get_model(model_path, dict_path)
    num_topics = lda.num_topics

    st.title("20 Newsgroups Topic Explorer")
    st.write("LDA model with %d topics." % num_topics)

    sidebar = st.sidebar
    sidebar.header("Topic")
    tid = sidebar.selectbox("topic id", list(range(num_topics)))
    top_n = sidebar.slider("top words", 5, 30, 12)

    st.subheader("Top words for topic %d" % tid)
    pairs = lda.show_topic(tid, topn=top_n)
    df = pd.DataFrame(pairs, columns=["word", "weight"])
    st.bar_chart(df.set_index("word"))

    coh_path = os.path.join(cfg["paths"]["artifacts_dir"], "coherence.json")
    if os.path.exists(coh_path):
        st.subheader("Coherence vs k")
        with open(coh_path) as f:
            blob = json.load(f)
        sweep = pd.DataFrame(blob["sweep"], columns=["k", "coherence"])
        st.line_chart(sweep.set_index("k"))

    st.subheader("Try a new document")
    text = st.text_area("paste some text", height=200)
    if text:
        out = predict_topics(text, lda, dictionary)
        if not out:
            st.write("No topic above the probability threshold.")
        else:
            rows = []
            for t, p in out:
                words = ", ".join(w for w, _ in lda.show_topic(t, topn=6))
                rows.append({"topic": t, "prob": round(p, 3), "top words": words})
            st.table(pd.DataFrame(rows))


if __name__ == "__main__":
    main()
</content>
</invoke>