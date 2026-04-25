# Dockerfile for Price Detection System

FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set the working directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p logs data models results

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Default command - inference API
CMD ["python", "-m", "uvicorn", "src.inference.api:app", "--host", "0.0.0.0", "--port", "8000"]

# To override with training:
# docker run --gpus all price-detection python scripts/train.py ...
# To override with inference:
# docker run --gpus all price-detection python scripts/infer.py ...
