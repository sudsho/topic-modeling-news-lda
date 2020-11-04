#!/usr/bin/env bash
# one-shot setup helper. creates a venv, installs deps, downloads spacy model
# and the small NLTK pieces we use.
set -euo pipefail

PYTHON=${PYTHON:-python3.8}

if [ ! -d ".venv" ]; then
    "$PYTHON" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

echo "done. activate with: source .venv/bin/activate"
</content>
</invoke>