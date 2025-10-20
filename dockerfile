# Utiliser une image de base Python
FROM python:3.12-slim

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers du projet
COPY . /app

# Installer les dépendances
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Exposer le port de l'API et MLflow UI
EXPOSE 8000 5000





# Démarre FastAPI et MLflow simultanément via un script shell
CMD ["bash", "-c", "mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000 & uvicorn app:app --host 0.0.0.0 --port 8000 --reload"]
