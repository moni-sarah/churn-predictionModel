# Customer Churn Prediction API - Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying the Customer Churn Prediction API in production environments. The API serves machine learning predictions for customer churn risk assessment.

## Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)
- Required dependencies (see requirements.txt)
- Trained model file (churn_model.joblib)

## Quick Start

### 1. Environment Setup

```bash
# Create and activate virtual environment
python3 -m venv churn_env
source churn_env/bin/activate  # On macOS/Linux
# or
churn_env\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Train and Save Model

```bash
# Run the complete training pipeline
python churn_prediction_complete.py
```

This will generate:
- `churn_model.joblib` - Trained model file
- Visualization files (*.png)
- Model evaluation reports

### 3. Start the API Server

```bash
# Start the Flask API server
python serve_model.py
```

The server will start on `http://localhost:5000`

### 4. Test the API

```bash
# In a new terminal, run the test suite
python test_api.py
```

## API Endpoints

### Health Check
```http
GET /health
```
Returns API status and model loading information.

### Model Information
```http
GET /model/info
```
Returns details about the loaded model.

### Single Prediction
```http
POST /predict
Content-Type: application/json

{
    "Tenure": 5,
    "MonthlyCharges": 70.0,
    "TotalCharges": 350.0,
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check"
}
```

### Batch Prediction
```http
POST /predict/batch
Content-Type: application/json

{
    "customers": [
        {
            "Tenure": 5,
            "MonthlyCharges": 70.0,
            "TotalCharges": 350.0,
            "Contract": "Month-to-month",
            "PaymentMethod": "Electronic check"
        }
    ]
}
```

## Production Deployment

### 1. Using Gunicorn (Recommended)

```bash
# Install Gunicorn
pip install gunicorn

# Start with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 serve_model:app
```

### 2. Using Docker

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "serve_model:app"]
```

Build and run:

```bash
docker build -t churn-prediction-api .
docker run -p 5000:5000 churn-prediction-api
```

### 3. Using Systemd (Linux)

Create `/etc/systemd/system/churn-api.service`:

```ini
[Unit]
Description=Customer Churn Prediction API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/churn_prediction
Environment=PATH=/path/to/churn_env/bin
ExecStart=/path/to/churn_env/bin/gunicorn -w 4 -b 0.0.0.0:5000 serve_model:app
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable churn-api
sudo systemctl start churn-api
```

## Configuration

### Environment Variables

Set these environment variables for production:

```bash
export FLASK_ENV=production
export MODEL_PATH=/path/to/churn_model.joblib
export LOG_LEVEL=INFO
```

### API Configuration

Modify `serve_model.py` for production settings:

```python
# Production settings
app.config['DEBUG'] = False
app.config['TESTING'] = False

# Security headers
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response
```

## Monitoring and Logging

### 1. Application Logging

The API includes built-in logging. Configure log levels:

```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
```

### 2. Health Monitoring

Implement health checks:

```bash
# Check API health
curl http://localhost:5000/health

# Monitor response time
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:5000/health
```

### 3. Metrics Collection

Add Prometheus metrics:

```python
from prometheus_flask_exporter import PrometheusMetrics

metrics = PrometheusMetrics(app)
metrics.info('churn_prediction_api', 'Customer Churn Prediction API')
```

## Security Considerations

### 1. Authentication

Add API key authentication:

```python
from functools import wraps
from flask import request, jsonify

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key != 'your-secret-key':
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/predict', methods=['POST'])
@require_api_key
def predict_churn():
    # ... existing code
```

### 2. Rate Limiting

Implement rate limiting:

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict_churn():
    # ... existing code
```

### 3. Input Validation

Enhance input validation:

```python
from marshmallow import Schema, fields, validate

class CustomerSchema(Schema):
    Tenure = fields.Integer(required=True, validate=validate.Range(min=0, max=100))
    MonthlyCharges = fields.Float(required=True, validate=validate.Range(min=0))
    TotalCharges = fields.Float(required=True, validate=validate.Range(min=0))
    Contract = fields.String(required=True, validate=validate.OneOf(['Month-to-month', 'One year', 'Two year']))
    PaymentMethod = fields.String(required=True)
```

## Performance Optimization

### 1. Caching

Implement Redis caching:

```python
import redis
from functools import wraps

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_result(expire_time=3600):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            cache_key = f"prediction:{hash(str(args) + str(kwargs))}"
            result = redis_client.get(cache_key)
            if result:
                return json.loads(result)
            result = f(*args, **kwargs)
            redis_client.setex(cache_key, expire_time, json.dumps(result))
            return result
        return decorated_function
    return decorator
```

### 2. Load Balancing

Use Nginx as a reverse proxy:

```nginx
upstream churn_api {
    server 127.0.0.1:5000;
    server 127.0.0.1:5001;
    server 127.0.0.1:5002;
    server 127.0.0.1:5003;
}

server {
    listen 80;
    server_name api.yourcompany.com;

    location / {
        proxy_pass http://churn_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 3. Database Integration

For high-volume predictions, consider database storage:

```python
import sqlite3

def store_prediction(customer_data, prediction_result):
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions (customer_data, prediction, probability, timestamp)
        VALUES (?, ?, ?, ?)
    ''', (json.dumps(customer_data), prediction_result['churn_prediction'], 
          prediction_result['churn_probability'], datetime.now()))
    conn.commit()
    conn.close()
```

## Troubleshooting

### Common Issues

1. **Model Loading Error**
   ```
   ERROR: Failed to load model
   ```
   - Ensure `churn_model.joblib` exists
   - Check file permissions
   - Verify model compatibility

2. **Memory Issues**
   ```
   MemoryError: Unable to allocate array
   ```
   - Reduce batch size
   - Increase server memory
   - Use model compression

3. **Slow Response Times**
   - Enable caching
   - Optimize preprocessing
   - Use load balancing

### Debug Mode

Enable debug mode for development:

```python
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

## Integration Examples

### 1. CRM Integration

```python
import requests

def predict_customer_churn(customer_id, customer_data):
    response = requests.post(
        'http://api.yourcompany.com/predict',
        json=customer_data,
        headers={'X-API-Key': 'your-secret-key'}
    )
    
    if response.status_code == 200:
        result = response.json()
        # Update CRM with prediction
        update_crm_record(customer_id, result)
        return result
    else:
        raise Exception(f"API Error: {response.text}")
```

### 2. Webhook Integration

```python
@app.route('/webhook/prediction', methods=['POST'])
def webhook_prediction():
    customer_data = request.json
    prediction = predict_churn(customer_data)
    
    # Send webhook notification
    send_webhook_notification(prediction)
    
    return jsonify({'status': 'success'})
```

## Maintenance

### 1. Model Updates

```bash
# Retrain model
python churn_prediction_complete.py

# Restart API server
sudo systemctl restart churn-api
```

### 2. Backup Strategy

```bash
# Backup model files
cp churn_model.joblib backup/churn_model_$(date +%Y%m%d).joblib

# Backup logs
cp api.log backup/api_$(date +%Y%m%d).log
```

### 3. Performance Monitoring

```bash
# Monitor API performance
watch -n 5 'curl -s http://localhost:5000/health | jq'

# Check resource usage
htop
```

## Support

For technical support:
- Check logs: `tail -f api.log`
- Monitor health: `curl http://localhost:5000/health`
- Review documentation: README.md

---

**Note**: This deployment guide covers the essential aspects of production deployment. Adjust configurations based on your specific infrastructure and requirements. 