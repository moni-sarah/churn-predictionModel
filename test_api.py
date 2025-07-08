"""
Test script for the Customer Churn Prediction API
Demonstrates how to interact with the Flask API endpoints

Usage:
    python test_api.py
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:5000"

def test_health_check():
    """Test the health check endpoint."""
    print("=" * 50)
    print("Testing Health Check Endpoint")
    print("=" * 50)
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to API. Make sure the server is running.")
        return False

def test_model_info():
    """Test the model info endpoint."""
    print("\n" + "=" * 50)
    print("Testing Model Info Endpoint")
    print("=" * 50)
    
    try:
        response = requests.get(f"{BASE_URL}/model/info")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to API.")
        return False

def test_single_prediction():
    """Test single customer prediction."""
    print("\n" + "=" * 50)
    print("Testing Single Customer Prediction")
    print("=" * 50)
    
    # Test customer data
    customer_data = {
        "Tenure": 5,
        "MonthlyCharges": 70.0,
        "TotalCharges": 350.0,
        "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check"
    }
    
    print(f"Customer Data: {json.dumps(customer_data, indent=2)}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=customer_data,
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to API.")
        return False

def test_batch_prediction():
    """Test batch customer prediction."""
    print("\n" + "=" * 50)
    print("Testing Batch Customer Prediction")
    print("=" * 50)
    
    # Test batch data
    batch_data = {
        "customers": [
            {
                "Tenure": 5,
                "MonthlyCharges": 70.0,
                "TotalCharges": 350.0,
                "Contract": "Month-to-month",
                "PaymentMethod": "Electronic check"
            },
            {
                "Tenure": 10,
                "MonthlyCharges": 85.5,
                "TotalCharges": 850.5,
                "Contract": "Two year",
                "PaymentMethod": "Mailed check"
            },
            {
                "Tenure": 3,
                "MonthlyCharges": 55.3,
                "TotalCharges": 165.9,
                "Contract": "One year",
                "PaymentMethod": "Electronic check"
            }
        ]
    }
    
    print(f"Batch Data: {json.dumps(batch_data, indent=2)}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict/batch",
            json=batch_data,
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to API.")
        return False

def test_error_handling():
    """Test error handling with invalid data."""
    print("\n" + "=" * 50)
    print("Testing Error Handling")
    print("=" * 50)
    
    # Test with missing fields
    invalid_data = {
        "Tenure": 5,
        "MonthlyCharges": 70.0
        # Missing required fields
    }
    
    print(f"Invalid Data (missing fields): {json.dumps(invalid_data, indent=2)}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=invalid_data,
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 400
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to API.")
        return False

def main():
    """Run all tests."""
    print("Customer Churn Prediction API Test Suite")
    print("=" * 60)
    print("Make sure the API server is running on http://localhost:5000")
    print("=" * 60)
    
    # Wait a moment for server to be ready
    time.sleep(1)
    
    # Run tests
    tests = [
        ("Health Check", test_health_check),
        ("Model Info", test_model_info),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Error Handling", test_error_handling)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name} test...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"ERROR in {test_name}: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! API is working correctly.")
    else:
        print("❌ Some tests failed. Check the API server and model file.")

if __name__ == "__main__":
    main() 