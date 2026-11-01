# API image. Choose the model backend at build time:
#   docker build -t psoriscan-api .                      # PyTorch ResNet18 (default, smaller)
#   docker build --build-arg BACKEND=tf -t psoriscan-api .
# The model file is not baked in: set MODEL_URL (https link to the file) so the container
# downloads it on first start, or mount a folder containing it at /app/models.
ARG BACKEND=torch

FROM python:3.11-slim
ARG BACKEND
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_BACKEND=${BACKEND}

WORKDIR /app
COPY requirements.txt requirements-torch.txt requirements-tf.txt ./
RUN pip install -r requirements.txt \
 && if [ "$BACKEND" = "tf" ]; then \
      pip install -r requirements-tf.txt; \
    else \
      pip install --index-url https://download.pytorch.org/whl/cpu -r requirements-torch.txt; \
    fi

COPY psoriscan ./psoriscan
COPY app.py .
RUN useradd --create-home appuser && mkdir -p /app/models && chown appuser /app/models
USER appuser

# Platforms such as Render, Cloud Run and Hugging Face Spaces set $PORT.
ENV PORT=5000
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import os,urllib.request as u; u.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health')"

# One worker keeps a single copy of the model in memory; threads handle concurrency.
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 2 --timeout 120 'app:create_app()'"]
