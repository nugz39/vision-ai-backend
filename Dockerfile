FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# OS deps
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt

# CPU torch first, then the rest
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# Default port; Spaces will inject PORT=7860 automatically
ENV PORT=8000
EXPOSE 8000

CMD ["sh","-lc","python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
