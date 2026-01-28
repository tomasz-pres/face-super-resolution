# =============================================================================
# Face Super-Resolution - Docker Configuration
# =============================================================================

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies (noninteractive to avoid timezone prompts)
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary application files
COPY src/ ./src/
COPY app/ ./app/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Copy Docker-specific files
COPY docker-compose.yml .
COPY Dockerfile .

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Note: checkpoints/ will be mounted as volume at runtime

# Expose Gradio port
EXPOSE 7860

# Default command
CMD ["python", "app/demo.py"]
