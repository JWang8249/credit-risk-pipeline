train:
	python src/model_train.py

eval:
	python src/model_eval.py

serve:
	python -m uvicorn src.api:app --reload --port 8000

app:
	streamlit run src/app.py

test:
	pytest -q
