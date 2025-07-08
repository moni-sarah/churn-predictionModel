"""
Generate Larger Synthetic Customer Churn Dataset
Creates a realistic dataset for training the churn prediction model
"""

import pandas as pd
import numpy as np
import random

def generate_synthetic_churn_data(n_customers=1000):
    """
    Generate synthetic customer churn data for training.
    
    Args:
        n_customers (int): Number of customers to generate
        
    Returns:
        pd.DataFrame: Synthetic customer data
    """
    
    np.random.seed(42)
    random.seed(42)
    
    # Customer IDs
    customer_ids = range(1001, 1001 + n_customers)
    
    # Tenure (months) - most customers have shorter tenure
    tenure = np.random.exponential(scale=12, size=n_customers)
    tenure = np.clip(tenure, 1, 72).astype(int)
    
    # Monthly charges - normal distribution with some outliers
    monthly_charges = np.random.normal(65, 20, n_customers)
    monthly_charges = np.clip(monthly_charges, 20, 150)
    
    # Total charges - based on tenure and monthly charges
    total_charges = tenure * monthly_charges * np.random.uniform(0.8, 1.2, n_customers)
    
    # Contract types with realistic distribution
    contracts = ['Month-to-month', 'One year', 'Two year']
    contract_weights = [0.5, 0.3, 0.2]  # More month-to-month customers
    contract = np.random.choice(contracts, n_customers, p=contract_weights)
    
    # Payment methods
    payment_methods = ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card']
    payment_method = np.random.choice(payment_methods, n_customers)
    
    # Generate churn based on realistic patterns
    churn_prob = np.zeros(n_customers)
    
    for i in range(n_customers):
        # Base churn probability
        base_prob = 0.2
        
        # Higher churn for month-to-month contracts
        if contract[i] == 'Month-to-month':
            base_prob += 0.3
        elif contract[i] == 'One year':
            base_prob += 0.1
        else:  # Two year
            base_prob -= 0.1
        
        # Higher churn for shorter tenure
        if tenure[i] < 6:
            base_prob += 0.2
        elif tenure[i] > 24:
            base_prob -= 0.1
        
        # Higher churn for higher monthly charges
        if monthly_charges[i] > 80:
            base_prob += 0.1
        
        # Higher churn for electronic check payment
        if payment_method[i] == 'Electronic check':
            base_prob += 0.1
        
        # Add some randomness
        base_prob += np.random.normal(0, 0.05)
        base_prob = np.clip(base_prob, 0, 1)
        
        churn_prob[i] = base_prob
    
    # Generate churn outcomes
    churn = (np.random.random(n_customers) < churn_prob).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'CustomerID': customer_ids,
        'Tenure': tenure,
        'MonthlyCharges': monthly_charges.round(2),
        'TotalCharges': total_charges.round(2),
        'Contract': contract,
        'PaymentMethod': payment_method,
        'Churn': churn
    })
    
    return df

if __name__ == "__main__":
    print("Generating synthetic customer churn dataset...")
    
    # Generate 1000 customers
    df = generate_synthetic_churn_data(1000)
    
    # Save to CSV
    df.to_csv('customer_churn.csv', index=False)
    
    print(f"Dataset generated with {len(df)} customers")
    print(f"Churn rate: {df['Churn'].mean():.2%}")
    print(f"Dataset saved to customer_churn.csv")
    
    # Show sample
    print("\nSample data:")
    print(df.head(10))
    
    # Show statistics
    print("\nDataset statistics:")
    print(df.describe())
    
    print("\nChurn distribution:")
    print(df['Churn'].value_counts())
    
    print("\nContract distribution:")
    print(df['Contract'].value_counts())
    
    print("\nPayment method distribution:")
    print(df['PaymentMethod'].value_counts()) 