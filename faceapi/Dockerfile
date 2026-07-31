# Face Recognition API — CPU image
FROM python:3.12-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt requests

COPY faceapi/ ./faceapi/

# fetch models at build time into work/models (mount at runtime to skip)
RUN python -m faceapi.download_models || echo "models will be mounted at runtime"

ENV FACEAPI_HOST=0.0.0.0 FACEAPI_PORT=8080
EXPOSE 8080
CMD ["python", "-m", "faceapi.serve"]
