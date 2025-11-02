# Real-Time Object Detection Dockerfile
# Multi-stage build for optimized production image
# Fixed for Debian Trixie (python:3.11-slim) with correct Mesa packages and build-time downloads

# Build stage
FROM python:3.11-slim AS builder

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps for building and Ultralytics downloader (curl)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Production stage
FROM python:3.11-slim AS production

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_ENV=production \
    HOST=0.0.0.0 \
    PORT=5000

# Runtime system deps + curl for model downloads
RUN apt-get update && apt-get install -y \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user with home dir
RUN groupadd -r appuser && useradd -r -g appuser -m -d /home/appuser appuser

# Workdir
WORKDIR /app

# Copy virtualenv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# App files
COPY --chown=appuser:appuser app.py .
COPY --chown=appuser:appuser detect.py .
COPY --chown=appuser:appuser templates/ templates/

# Writable dirs and user config/cache to avoid warnings
RUN mkdir -p /home/appuser/.config/matplotlib /home/appuser/.config/Ultralytics /home/appuser/.cache/torch uploads models logs \
  && chown -R appuser:appuser /home/appuser/.config /home/appuser/.cache uploads models logs

ENV MPLCONFIGDIR=/home/appuser/.config/matplotlib \
    YOLO_CONFIG_DIR=/home/appuser/.config/Ultralytics \
    TORCH_HOME=/home/appuser/.cache/torch

# Switch to non-root
USER appuser

# Optional: pre-download YOLO model to warm cache (can be removed if flaky)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health', timeout=5).raise_for_status()" || exit 1

# Expose
EXPOSE 5000

# Start
CMD ["python", "app.py"]
