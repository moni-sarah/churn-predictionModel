
# Docker Deployment Guide for Customer Churn Prediction API

## Overview

This guide provides step-by-step instructions for containerizing and deploying the Customer Churn Prediction API using Docker.

## Prerequisites

- Docker installed on your system
- Docker Compose installed
- At least 2GB of available disk space

## Quick Start

### 1. Build and Run with Docker Compose

```bash
# Make the build script executable
chmod +x docker_build.sh

# Build and run the application
./docker_build.sh
```

### 2. Manual Docker Commands

```bash
# Build the Docker image
docker build -t churn-prediction-api .

# Run the container
docker run -p 5000:5000 churn-prediction-api
```

### 3. Using Docker Compose

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Dockerfile Explanation

### Multi-Stage Build

The Dockerfile uses a multi-stage build approach:

1. **Builder Stage**: Installs dependencies and compiles packages
2. **Production Stage**: Creates a minimal production image

### Security Features

- Non-root user (`appuser`)
- Minimal base image (`python:3.9-slim`)
- No unnecessary packages
- Health checks included

### Environment Variables

```dockerfile
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV FLASK_ENV=production
ENV PORT=5000
```

## Docker Compose Configuration

### Services

- **churn-prediction-api**: Main API service
- **redis** (optional): Caching service
- **nginx** (optional): Reverse proxy

### Volumes

- `./models:/app/models`: Model storage
- `./logs:/app/logs`: Application logs

### Networks

- `churn-network`: Isolated network for services

## Testing the Docker Container

### 1. Health Check

```bash
curl http://localhost:5000/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00",
  "model_loaded": true,
  "service": "Customer Churn Prediction API",
  "version": "1.0.0"
}
```

### 2. Model Information

```bash
curl http://localhost:5000/model/info
```

### 3. Single Prediction

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Tenure": 5,
    "MonthlyCharges": 70.0,
    "TotalCharges": 350.0,
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check"
  }'
```

### 4. Batch Prediction

```bash
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [
      {
        "Tenure": 5,
        "MonthlyCharges": 70.0,
        "TotalCharges": 350.0,
        "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check"
      }
    ]
  }'
```

### 5. Automated Testing

```bash
# Run the comprehensive test suite
python test_docker_api.py
```

## Production Deployment

### 1. Environment Variables

Create a `.env` file:

```env
FLASK_ENV=production
PYTHONUNBUFFERED=1
MODEL_PATH=/app/models/churn_model.joblib
LOG_LEVEL=INFO
```

### 2. Production Docker Compose

```yaml
version: '3.8'

services:
  churn-prediction-api:
    build: .
    container_name: churn-prediction-api-prod
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - churn-network

  nginx:
    image: nginx:alpine
    container_name: churn-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - churn-prediction-api
    restart: unless-stopped
    networks:
      - churn-network

networks:
  churn-network:
    driver: bridge
```

### 3. Nginx Configuration

Create `nginx.conf`:

```nginx
events {
    worker_connections 1024;
}

http {
    upstream churn_api {
        server churn-prediction-api:5000;
    }

    server {
        listen 80;
        server_name localhost;

        location / {
            proxy_pass http://churn_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

## Monitoring and Logging

### 1. View Container Logs

```bash
# View logs for specific service
docker-compose logs churn-prediction-api

# Follow logs in real-time
docker-compose logs -f churn-prediction-api

# View logs for all services
docker-compose logs
```

### 2. Container Health

```bash
# Check container status
docker-compose ps

# Check container health
docker inspect churn-prediction-api | grep Health -A 10
```

### 3. Resource Usage

```bash
# Monitor resource usage
docker stats churn-prediction-api

# Check disk usage
docker system df
```

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Check what's using port 5000
   lsof -i :5000
   
   # Change port in docker-compose.yml
   ports:
     - "5001:5000"
   ```

2. **Model Not Found**
   ```bash
   # Ensure model file exists
   ls -la churn_model.joblib
   
   # Rebuild if needed
   docker-compose down
   docker-compose build --no-cache
   docker-compose up -d
   ```

3. **Permission Issues**
   ```bash
   # Fix file permissions
   chmod 755 docker_build.sh
   chmod 644 churn_model.joblib
   ```

4. **Memory Issues**
   ```bash
   # Increase Docker memory limit
   # In Docker Desktop: Settings > Resources > Memory
   ```

### Debug Mode

```bash
# Run container in interactive mode
docker run -it --rm -p 5000:5000 churn-prediction-api /bin/bash

# Check container filesystem
docker exec -it churn-prediction-api ls -la /app
```

## Scaling

### 1. Horizontal Scaling

```bash
# Scale to multiple instances
docker-compose up -d --scale churn-prediction-api=3
```

### 2. Load Balancing

Update nginx configuration for multiple instances:

```nginx
upstream churn_api {
    server churn-prediction-api:5000;
    server churn-prediction-api:5001;
    server churn-prediction-api:5002;
}
```

## Security Considerations

### 1. Image Security

```bash
# Scan for vulnerabilities
docker scan churn-prediction-api

# Use specific base image versions
FROM python:3.9.18-slim
```

### 2. Network Security

```bash
# Use custom network
docker network create churn-network

# Restrict container communication
docker run --network churn-network --network-alias api churn-prediction-api
```

### 3. Secrets Management

```bash
# Use Docker secrets for sensitive data
echo "my-secret-key" | docker secret create api_key -

# Reference in docker-compose.yml
secrets:
  - api_key
```

## Performance Optimization

### 1. Image Size

```bash
# Check image size
docker images churn-prediction-api

# Optimize with multi-stage build
# (Already implemented in Dockerfile)
```

### 2. Caching

```bash
# Use Docker layer caching
docker build --cache-from churn-prediction-api -t churn-prediction-api .
```

### 3. Resource Limits

```yaml
services:
  churn-prediction-api:
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M
```

## CI/CD Integration

### 1. GitHub Actions

```yaml
name: Build and Deploy

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build Docker image
        run: docker build -t churn-prediction-api .
      - name: Test Docker image
        run: docker run -d -p 5000:5000 churn-prediction-api
      - name: Run tests
        run: python test_docker_api.py
```

### 2. Docker Registry

```bash
# Tag and push to registry
docker tag churn-prediction-api your-registry/churn-prediction-api:latest
docker push your-registry/churn-prediction-api:latest
```

## Backup and Recovery

### 1. Backup Model

```bash
# Backup model file
docker cp churn-prediction-api:/app/models/churn_model.joblib ./backup/

# Backup logs
docker cp churn-prediction-api:/app/logs ./backup/
```

### 2. Restore

```bash
# Restore from backup
docker cp ./backup/churn_model.joblib churn-prediction-api:/app/models/
docker restart churn-prediction-api
```

---

**Note**: This Docker setup provides a production-ready containerized deployment of the Customer Churn Prediction API. Regular monitoring and updates are essential for maintaining performance and security. 