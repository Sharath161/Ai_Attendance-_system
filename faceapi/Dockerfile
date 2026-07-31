# Face Recognition API — CPU image
FROM python:3.12-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*

COPY requirements-faceapi.txt .
RUN pip install --no-cache-dir -r requirements-faceapi.txt

# service code + the low-level model helpers it reuses
COPY faceapi/ ./faceapi/
COPY worker/download_models.py ./worker/download_models.py
COPY worker/__init__.py ./worker/__init__.py
COPY core/ ./core/

# fetch models at build time into work/models
RUN python -m worker.download_models || echo "models will be mounted at runtime"

ENV FACEAPI_HOST=0.0.0.0 FACEAPI_PORT=8080
EXPOSE 8080
CMD ["python", "-m", "faceapi.serve"]
