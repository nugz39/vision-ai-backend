# Use slim image for smaller/faster builds
FROM python:3.11-slim

# Avoid debconf TTY prompts and speed up apt
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/hf \
    HUGGINGFACE_HUB_CACHE=/cache/hf \
    HF_HUB_ENABLE_HF_TRANSFER=1

# System deps (git for models, ffmpeg for video/enc, curl for health/debug)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg curl && \
    rm -rf /var/lib/apt/lists/*

# Make cache dir world-writable so Space runner can populate it
RUN mkdir -p /cache/hf && chmod -R 777 /cache

WORKDIR /app

# Layered dependency install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Healthcheck (Space honors PORT)
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=5 \
  CMD curl -fsS http://127.0.0.1:${PORT:-7860}/healthz || exit 1

# Start server
CMD ["bash", "start.sh"]
