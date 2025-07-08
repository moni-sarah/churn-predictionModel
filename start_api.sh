#!/bin/bash

# Customer Churn Prediction API Startup Script
# This script sets up the environment and starts the API server

echo "=========================================="
echo "Customer Churn Prediction API"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d "churn_env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv churn_env
fi

# Activate virtual environment
echo "Activating virtual environment..."
source churn_env/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if model exists
if [ ! -f "churn_model.joblib" ]; then
    echo "Training model..."
    python churn_prediction_complete.py
fi

# Start the API server
echo "Starting API server..."
echo "API will be available at: http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo "=========================================="

python serve_model.py 