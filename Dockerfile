# Dockerfile for FastAPI backend on Railway
FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip && pip install -r requirements.txt

CMD ["uvicorn", "auto_ml.deployment.api.simple_api:app", "--host", "0.0.0.0", "--port", "8000"] 