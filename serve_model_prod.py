"""
Customer Churn Prediction API - Production Version
Optimized for Docker deployment with enhanced logging and error handling

Author: Data Science Team
Date: 2024
"""

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import logging
import os
import sys
from datetime import datetime
import traceback

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/api.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for model components
model = None
scaler = None
label_encoders = None
feature_names = None

def load_model(model_path='/app/models/churn_model.joblib'):
    """
    Load the trained model and preprocessing components.
    
    Args:
        model_path (str): Path to the saved model file
        
    Returns:
        bool: True if model loaded successfully, False otherwise
    """
    global model, scaler, label_encoders, feature_names
    
    try:
        # Check if model file exists
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False
            
        # Load model components
        logger.info(f"Loading model from {model_path}")
        model_components = joblib.load(model_path)
        
        model = model_components['model']
        scaler = model_components['scaler']
        label_encoders = model_components['label_encoders']
        feature_names = model_components['feature_names']
        
        logger.info(f"Model loaded successfully")
        logger.info(f"Model type: {type(model).__name__}")
        logger.info(f"Number of features: {len(feature_names)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def preprocess_input(customer_data):
    """
    Preprocess customer data for prediction.
    
    Args:
        customer_data (dict): Raw customer data
        
    Returns:
        numpy.ndarray: Preprocessed feature array
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame([customer_data])
        
        # Apply label encoding for categorical variables
        for col in label_encoders:
            if col in df.columns:
                # Handle unseen categories
                unique_values = df[col].unique()
                for val in unique_values:
                    if val not in label_encoders[col].classes_:
                        # Map to most common category or default
                        df[col] = df[col].replace(val, label_encoders[col].classes_[0])
                
                df[col] = label_encoders[col].transform(df[col])
        
        # Ensure all required features are present
        for feature in feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        # Reorder columns to match training data
        df = df[feature_names]
        
        # Scale features
        df_scaled = scaler.transform(df)
        
        return df_scaled
        
    except Exception as e:
        logger.error(f"Error preprocessing input: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify API is running.
    
    Returns:
        JSON: API status information
    """
    try:
        status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'model_loaded': model is not None,
            'service': 'Customer Churn Prediction API',
            'version': '1.0.0'
        }
        
        if model is not None:
            status['model_type'] = type(model).__name__
            status['features_count'] = len(feature_names) if feature_names else 0
        
        logger.info("Health check requested")
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """
    Get information about the loaded model.
    
    Returns:
        JSON: Model information
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        info = {
            'model_type': type(model).__name__,
            'feature_names': feature_names,
            'categorical_features': list(label_encoders.keys()) if label_encoders else [],
            'loaded_at': datetime.now().isoformat(),
            'model_parameters': str(model.get_params()) if hasattr(model, 'get_params') else 'Not available'
        }
        
        logger.info("Model info requested")
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"Model info failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict_churn():
    """
    Make churn prediction for a customer.
    
    Expected JSON input:
    {
        "Tenure": 5,
        "MonthlyCharges": 70.0,
        "TotalCharges": 350.0,
        "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check"
    }
    
    Returns:
        JSON: Prediction results
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get input data
        customer_data = request.get_json()
        
        if not customer_data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['Tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'PaymentMethod']
        missing_fields = [field for field in required_fields if field not in customer_data]
        
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {missing_fields}',
                'required_fields': required_fields
            }), 400
        
        # Preprocess input
        X_scaled = preprocess_input(customer_data)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1]
        
        # Prepare response
        response = {
            'customer_data': customer_data,
            'prediction': {
                'churn_prediction': bool(prediction),
                'churn_probability': float(probability),
                'retention_probability': float(1 - probability)
            },
            'risk_level': get_risk_level(probability),
            'recommendations': get_recommendations(probability, customer_data),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Prediction made: churn_prob={probability:.3f}, risk={response['risk_level']}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Make churn predictions for multiple customers.
    
    Expected JSON input:
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
    
    Returns:
        JSON: Batch prediction results
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get input data
        data = request.get_json()
        
        if not data or 'customers' not in data:
            return jsonify({'error': 'No customers data provided'}), 400
        
        customers = data['customers']
        
        if not isinstance(customers, list):
            return jsonify({'error': 'Customers must be a list'}), 400
        
        if len(customers) > 1000:  # Limit batch size
            return jsonify({'error': 'Batch size too large. Maximum 1000 customers.'}), 400
        
        results = []
        successful = 0
        failed = 0
        
        for i, customer_data in enumerate(customers):
            try:
                # Validate required fields
                required_fields = ['Tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'PaymentMethod']
                missing_fields = [field for field in required_fields if field not in customer_data]
                
                if missing_fields:
                    results.append({
                        'customer_id': i,
                        'error': f'Missing required fields: {missing_fields}'
                    })
                    failed += 1
                    continue
                
                # Preprocess input
                X_scaled = preprocess_input(customer_data)
                
                # Make prediction
                prediction = model.predict(X_scaled)[0]
                probability = model.predict_proba(X_scaled)[0][1]
                
                results.append({
                    'customer_id': i,
                    'customer_data': customer_data,
                    'prediction': {
                        'churn_prediction': bool(prediction),
                        'churn_probability': float(probability),
                        'retention_probability': float(1 - probability)
                    },
                    'risk_level': get_risk_level(probability),
                    'recommendations': get_recommendations(probability, customer_data)
                })
                successful += 1
                
            except Exception as e:
                results.append({
                    'customer_id': i,
                    'error': f'Prediction failed: {str(e)}'
                })
                failed += 1
        
        response = {
            'total_customers': len(customers),
            'successful_predictions': successful,
            'failed_predictions': failed,
            'predictions': results,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Batch prediction completed: {successful}/{len(customers)} successful")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500

def get_risk_level(probability):
    """
    Determine risk level based on churn probability.
    
    Args:
        probability (float): Churn probability (0-1)
        
    Returns:
        str: Risk level
    """
    if probability < 0.3:
        return 'LOW'
    elif probability < 0.6:
        return 'MEDIUM'
    else:
        return 'HIGH'

def get_recommendations(probability, customer_data):
    """
    Generate retention recommendations based on prediction and customer data.
    
    Args:
        probability (float): Churn probability
        customer_data (dict): Customer information
        
    Returns:
        list: List of recommendations
    """
    recommendations = []
    
    # High risk customers
    if probability > 0.6:
        recommendations.append("URGENT: High churn risk detected")
        recommendations.append("Immediate retention intervention required")
        
        # Contract-specific recommendations
        if customer_data.get('Contract') == 'Month-to-month':
            recommendations.append("Offer long-term contract with discount")
        elif customer_data.get('Contract') == 'One year':
            recommendations.append("Consider upgrading to two-year contract")
    
    # Medium risk customers
    elif probability > 0.3:
        recommendations.append("Moderate churn risk - proactive outreach recommended")
        recommendations.append("Offer personalized retention incentives")
    
    # Low risk customers
    else:
        recommendations.append("Low churn risk - maintain current service quality")
        recommendations.append("Focus on upselling opportunities")
    
    # Tenure-based recommendations
    tenure = customer_data.get('Tenure', 0)
    if tenure < 6:
        recommendations.append("New customer - focus on onboarding and satisfaction")
    elif tenure > 24:
        recommendations.append("Long-term customer - leverage loyalty programs")
    
    # Payment method recommendations
    payment_method = customer_data.get('PaymentMethod', '')
    if 'Electronic check' in payment_method:
        recommendations.append("Consider offering automatic payment discount")
    
    return recommendations

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500

@app.before_request
def log_request():
    """Log all incoming requests."""
    logger.info(f"{request.method} {request.path} - {request.remote_addr}")

if __name__ == '__main__':
    # Load model on startup
    logger.info("Starting Customer Churn Prediction API...")
    
    # Try multiple model paths
    model_paths = [
        '/app/models/churn_model.joblib',
        'churn_model.joblib',
        '/app/churn_model.joblib'
    ]
    
    model_loaded = False
    for model_path in model_paths:
        if load_model(model_path):
            model_loaded = True
            break
    
    if not model_loaded:
        logger.error("Failed to load model from any location")
        logger.error("Available model paths tried:")
        for path in model_paths:
            logger.error(f"  - {path}: {'EXISTS' if os.path.exists(path) else 'NOT FOUND'}")
        sys.exit(1)
    
    logger.info("Model loaded successfully!")
    logger.info("Starting Flask API server...")
    
    # Get port from environment variable
    port = int(os.environ.get('PORT', 5000))
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=port, debug=False) 