# Dockerfile for FastAPI backend on Railway
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for LightGBM and XGBoost
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --upgrade pip && pip install -r requirements.txt

CMD ["uvicorn", "auto_ml.deployment.api.simple_api:app", "--host", "0.0.0.0", "--port", "8000"] 