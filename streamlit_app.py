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
from src.topics import all_labels, short_label
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
    labels = all_labels(lda, n=3)
    options = ["%d: %s" % (i, lab) for i, lab in enumerate(labels)]
    selected = sidebar.selectbox("topic", options)
    tid = int(selected.split(":", 1)[0])
    top_n = sidebar.slider("top words", 5, 30, 12)

    st.subheader("Topic %d (%s)" % (tid, short_label(lda, tid, n=3)))
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

    st.subheader("Document explorer")
    samples_path = "data/samples/posts.json"
    if os.path.exists(samples_path):
        with open(samples_path) as f:
            samples = json.load(f)
        choice = st.selectbox(
            "pick a sample post",
            ["(none)"] + ["%d. %s" % (i, s["label"]) for i, s in enumerate(samples)],
        )
        if choice != "(none)":
            i = int(choice.split(".", 1)[0])
            sample = samples[i]
            st.markdown("**label**: `%s`" % sample["label"])
            st.write(sample["text"])
            out = predict_topics(sample["text"], lda, dictionary)
            rows = [{"topic": t, "prob": round(p, 3)} for t, p in out]
            st.write("predicted topics:")
            st.table(pd.DataFrame(rows))

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