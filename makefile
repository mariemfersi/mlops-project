VENV ?= venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
DATA_PATH ?= insurance.csv
TARGET ?= charges
MODEL_PATH ?= model.pkl
NOTEBOOK_PORT ?= 8888
API_PORT ?= 8000

.PHONY: setup prepare train evaluate save all clean notebook lint format test api

setup:
	@if [ ! -d "$(VENV)" ]; then \
		python3 -m venv $(VENV); \
	fi
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Virtual environment ready."

prepare:
	$(PYTHON) main.py --prepare --data_path $(DATA_PATH) --target $(TARGET)
	@echo "Données préparées."

train:
	$(PYTHON) main.py --train --prepare --data_path $(DATA_PATH) --target $(TARGET)
	@echo "Modèle entraîné."

evaluate:
	$(PYTHON) main.py --evaluate --prepare --train --data_path $(DATA_PATH) --target $(TARGET)
	@echo "Évaluation terminée."

save:
	$(PYTHON) main.py --save --prepare --train --data_path $(DATA_PATH) --target $(TARGET) --model_path $(MODEL_PATH)
	@echo "Modèle sauvegardé dans $(MODEL_PATH)."

all: setup prepare train evaluate save
	@echo "Pipeline complet exécuté."

notebook:
	$(PYTHON) -m notebook insurance.ipynb --port=$(NOTEBOOK_PORT) --no-browser

api:
	$(PYTHON) -m uvicorn app:app --reload --port $(API_PORT)

mlflow:
	mlflow ui --port 5000
	@echo "MLflow UI running at http://127.0.0.1:5000"


clean:
	rm -rf $(VENV) __pycache__ *.pyc *.pyo *.pkl *.png
	@echo "Nettoyage terminé."

lint:
	-$(PYTHON) -m flake8 . --exclude=$(VENV),__pycache__

format:
	$(PYTHON) -m black .

test:
	$(PYTHON) test_environment.py
	@echo "Tests exécutés."

