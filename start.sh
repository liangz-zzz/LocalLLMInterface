#!/bin/bash

# Local LLM Interface Startup Script

set -e

echo "🚀 Starting Local LLM Interface"
echo "================================"

bind_ip="${LLM_BIND_IP:-127.0.0.1}"
base_url="http://${bind_ip}:15530"
health_url="${base_url}/v1/health"

curl_args=(-s)
if [ -n "${LLM_API_KEY:-}" ]; then
    curl_args+=(-H "Authorization: Bearer ${LLM_API_KEY}")
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if NVIDIA Container Toolkit is available
if ! docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu20.04 nvidia-smi > /dev/null 2>&1; then
    echo "⚠️  NVIDIA Container Toolkit not available. GPU acceleration may not work."
fi

# Check if models directory exists
if [ ! -d "/mnt/workspace/Source/LLM" ]; then
    echo "❌ Models directory /mnt/workspace/Source/LLM not found."
    echo "Please ensure you have downloaded models to this location."
    exit 1
fi

# Count available models
model_count=$(find /mnt/workspace/Source/LLM -maxdepth 1 -type d | wc -l)
echo "📁 Found $((model_count - 1)) model directories in /mnt/workspace/Source/LLM"

# Start the service (image already built)
echo "🚀 Starting service..."
docker-compose up -d

# Wait for service to be ready
echo "⏳ Waiting for service to start..."
sleep 10

# Test service
echo "🧪 Testing service..."
if curl "${curl_args[@]}" "${health_url}" > /dev/null; then
    echo "✅ Service is running successfully!"
    echo ""
    echo "📊 Service Information:"
    curl "${curl_args[@]}" "${health_url}" | python3 -m json.tool
    echo ""
    echo "🌐 Access the service at:"
    echo "  API Base URL: ${base_url}/v1"
    echo "  Documentation: ${base_url}/docs"
    echo "  Health Check: ${health_url}"
    echo "  Model List: ${base_url}/v1/models"
    echo ""
    echo "📖 Usage Examples:"
    echo "  Python: python test_api.py"
    if [ -n "${LLM_API_KEY:-}" ]; then
        echo "  cURL: curl -H 'Authorization: Bearer <LLM_API_KEY>' ${base_url}/v1/models"
    else
        echo "  cURL: curl ${base_url}/v1/models"
    fi
else
    echo "❌ Service failed to start. Check logs:"
    echo "docker-compose logs local-llm-interface"
    exit 1
fi
