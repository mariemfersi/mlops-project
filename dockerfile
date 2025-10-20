
FROM python:3.10-slim

# Empêche Python de créer des fichiers .pyc et active un buffering standard
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app


COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .



# 8000 : FastAPI
# 5000 : MLflow
EXPOSE 8000
EXPOSE 5000


# Démarre FastAPI et MLflow simultanément via un script shell
CMD ["bash", "-c", "mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000 & uvicorn app:app --host 0.0.0.0 --port 8000 --reload"]
