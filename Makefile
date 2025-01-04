install:
	pip install -r requirements.txt

train:
	python train.py

train_unsloth:
	python train_unsloth.py

inf:
	python inference.py

lint:
	ruff check . --fix --unsafe-fixes

mlflow:
	mlflow ui

chainlit:
	chainlit run app.py -w

format:
	ruff format .

watch:
	watch -n 1 nvidia-smi