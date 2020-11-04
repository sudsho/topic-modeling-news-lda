.PHONY: install train vis app test clean

install:
	bash setup.sh

train:
	python -m src.lda --config configs/default.yaml

vis:
	python -m src.visualize --config configs/default.yaml --out artifacts/lda_vis.html

app:
	streamlit run streamlit_app.py

test:
	pytest -q

clean:
	rm -rf artifacts/*.model artifacts/*.gensim artifacts/lda_vis.html
</content>
</invoke>