"""
Test script for Docker containerized Customer Churn Prediction API
Tests all endpoints and provides detailed feedback

Usage:
    python test_docker_api.py
"""

import requests
import json
import time
import sys

# API base URL (Docker container)
BASE_URL = "http://localhost:5000"

def wait_for_api(max_retries=30, delay=2):
    """Wait for the API to be ready."""
    print("Waiting for API to be ready...")
    
    for i in range(max_retries):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                print("✅ API is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        print(f"Attempt {i+1}/{max_retries}: API not ready yet, waiting {delay}s...")
        time.sleep(delay)
    
    print("❌ API failed to start within expected time")
    return False

def test_health_check():
    """Test the health check endpoint."""
    print("\n" + "=" * 50)
    print("Testing Health Check Endpoint")
    print("=" * 50)
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed!")
            print(f"Response: {json.dumps(data, indent=2)}")
            return True
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: {str(e)}")
        return False

def test_model_info():
    """Test the model info endpoint."""
    print("\n" + "=" * 50)
    print("Testing Model Info Endpoint")
    print("=" * 50)
    
    try:
        response = requests.get(f"{BASE_URL}/model/info", timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Model info retrieved successfully!")
            print(f"Model Type: {data.get('model_type', 'Unknown')}")
            print(f"Features: {len(data.get('feature_names', []))}")
            print(f"Response: {json.dumps(data, indent=2)}")
            return True
        else:
            print(f"❌ Model info failed with status {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Model info failed: {str(e)}")
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
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Single prediction successful!")
            print(f"Churn Probability: {data['prediction']['churn_probability']:.3f}")
            print(f"Risk Level: {data['risk_level']}")
            print(f"Recommendations: {len(data['recommendations'])} provided")
            print(f"Response: {json.dumps(data, indent=2)}")
            return True
        else:
            print(f"❌ Single prediction failed with status {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Single prediction failed: {str(e)}")
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
    
    print(f"Batch Data: {len(batch_data['customers'])} customers")
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict/batch",
            json=batch_data,
            headers={'Content-Type': 'application/json'},
            timeout=15
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Batch prediction successful!")
            print(f"Total Customers: {data['total_customers']}")
            print(f"Successful Predictions: {data['successful_predictions']}")
            print(f"Failed Predictions: {data['failed_predictions']}")
            
            # Show sample predictions
            for i, pred in enumerate(data['predictions'][:2]):
                if 'prediction' in pred:
                    print(f"  Customer {i}: Churn prob = {pred['prediction']['churn_probability']:.3f}")
            
            return True
        else:
            print(f"❌ Batch prediction failed with status {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Batch prediction failed: {str(e)}")
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
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 400:
            data = response.json()
            print("✅ Error handling working correctly!")
            print(f"Error: {data.get('error', 'Unknown error')}")
            return True
        else:
            print(f"❌ Expected 400 status, got {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Error handling test failed: {str(e)}")
        return False

def test_performance():
    """Test API performance with multiple requests."""
    print("\n" + "=" * 50)
    print("Testing API Performance")
    print("=" * 50)
    
    customer_data = {
        "Tenure": 5,
        "MonthlyCharges": 70.0,
        "TotalCharges": 350.0,
        "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check"
    }
    
    start_time = time.time()
    successful_requests = 0
    total_requests = 10
    
    for i in range(total_requests):
        try:
            response = requests.post(
                f"{BASE_URL}/predict",
                json=customer_data,
                headers={'Content-Type': 'application/json'},
                timeout=5
            )
            
            if response.status_code == 200:
                successful_requests += 1
            else:
                print(f"Request {i+1} failed with status {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"Request {i+1} failed: {str(e)}")
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / total_requests
    
    print(f"Performance Results:")
    print(f"  Total Requests: {total_requests}")
    print(f"  Successful: {successful_requests}")
    print(f"  Failed: {total_requests - successful_requests}")
    print(f"  Total Time: {total_time:.2f}s")
    print(f"  Average Time: {avg_time:.3f}s per request")
    
    if successful_requests == total_requests:
        print("✅ Performance test passed!")
        return True
    else:
        print("❌ Performance test failed!")
        return False

def main():
    """Run all tests."""
    print("Customer Churn Prediction API - Docker Test Suite")
    print("=" * 60)
    print("Make sure the Docker container is running")
    print("=" * 60)
    
    # Wait for API to be ready
    if not wait_for_api():
        print("❌ API is not accessible. Please check if the Docker container is running.")
        sys.exit(1)
    
    # Run tests
    tests = [
        ("Health Check", test_health_check),
        ("Model Info", test_model_info),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Error Handling", test_error_handling),
        ("Performance", test_performance)
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
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Docker API is working correctly.")
        print("The API is ready for production use!")
    else:
        print("❌ Some tests failed. Check the Docker container and logs.")
        print("To view logs: docker-compose logs -f")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 