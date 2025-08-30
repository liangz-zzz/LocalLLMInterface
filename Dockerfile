# Local LLM Interface Dockerfile
FROM nvcr.io/nvidia/pytorch:24.10-py3

# Set working directory
WORKDIR /app

### Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Create models directory (will be mounted from host)
RUN mkdir -p /models

# Note: Source code will be mounted via volume

# Set environment variables
ENV PYTHONPATH=/app
ENV LLM_MODELS_DIR=/models
ENV LLM_HOST=0.0.0.0
ENV LLM_PORT=15530

# Expose port
EXPOSE 15530

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:15530/v1/health || exit 1

# Start command
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "15530", "--workers", "1"]