"""tiny pickle cache so we don't re-tokenize the corpus every run."""
import hashlib
import os
import pickle


def cache_key(*parts):
    h = hashlib.sha1()
    for p in parts:
        h.update(repr(p).encode("utf-8"))
    return h.hexdigest()[:12]


def get(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def put(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path
</content>
</invoke>