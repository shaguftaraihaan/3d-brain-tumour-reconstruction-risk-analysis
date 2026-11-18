# Hugging Face Space (Docker SDK, CPU). Serves the Flask case explorer on port 7860.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    BRATS_DEMO=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch, then the rest of the app stack.
RUN pip install --upgrade pip \
 && pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Application code and public BraTS demo assets only (see .dockerignore).
COPY brats_recon.py ./
COPY configs ./configs
COPY recon ./recon
COPY scripts ./scripts
COPY web ./web
COPY splits ./splits
COPY data/cases/BraTS20_Training_002 ./data/cases/BraTS20_Training_002
COPY data/cases/BraTS20_Training_004 ./data/cases/BraTS20_Training_004
COPY data/cases/BraTS20_Training_022 ./data/cases/BraTS20_Training_022
COPY outputs/BraTS20_Training_002 ./outputs/BraTS20_Training_002
COPY outputs/BraTS20_Training_004 ./outputs/BraTS20_Training_004
COPY outputs/BraTS20_Training_022 ./outputs/BraTS20_Training_022

EXPOSE 7860

# One worker, long timeout: reconstruction is not enabled in the demo image,
# but volume/slice reads can still be slow on cold start.
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "300", "web.app:app"]
