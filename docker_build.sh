#!/bin/bash

# Docker Build and Run Script for Customer Churn Prediction API
# This script builds the Docker image and runs the container

echo "=========================================="
echo "Customer Churn Prediction API - Docker"
echo "=========================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "ERROR: docker-compose is not installed. Please install docker-compose first."
    exit 1
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p models logs

# Check if model exists
if [ ! -f "churn_model.joblib" ]; then
    echo "Model not found. Training model first..."
    source churn_env/bin/activate
    python churn_prediction_complete.py
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to train model"
        exit 1
    fi
fi

# Copy model to models directory
echo "Copying model to models directory..."
cp churn_model.joblib models/

# Build Docker image
echo "Building Docker image..."
docker build -t churn-prediction-api .

if [ $? -ne 0 ]; then
    echo "ERROR: Docker build failed"
    exit 1
fi

echo "Docker image built successfully!"

# Run with docker-compose
echo "Starting services with docker-compose..."
docker-compose up -d

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to start services"
    exit 1
fi

echo "=========================================="
echo "Services started successfully!"
echo "=========================================="
echo "API is running at: http://localhost:5000"
echo ""
echo "Available endpoints:"
echo "  GET  /health          - Health check"
echo "  GET  /model/info      - Model information"
echo "  POST /predict         - Single customer prediction"
echo "  POST /predict/batch   - Batch customer predictions"
echo ""
echo "To view logs:"
echo "  docker-compose logs -f"
echo ""
echo "To stop services:"
echo "  docker-compose down"
echo ""
echo "To test the API:"
echo "  curl http://localhost:5000/health"
echo "==========================================" 