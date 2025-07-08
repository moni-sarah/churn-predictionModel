"""
Customer Churn Prediction Model
Telecommunications Company - Machine Learning Solution

This script implements a complete machine learning pipeline for predicting customer churn.
The model is designed for production deployment in a CRM system.

Author: Data Science Team
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, f1_score
)
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class ChurnPredictionModel:
    """
    A comprehensive class for customer churn prediction.
    Handles data preprocessing, model training, evaluation, and deployment.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model = None
        self.feature_names = None
        
    def load_data(self, file_path):
        """Load and display basic information about the dataset."""
        print("=" * 60)
        print("STEP 1: LOADING DATASET")
        print("=" * 60)
        
        self.df = pd.read_csv(file_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        print("\nFirst 5 rows:")
        print(self.df.head())
        print("\nDataset info:")
        print(self.df.info())
        print("\nBasic statistics:")
        print(self.df.describe())
        
        return self.df
    
    def explore_data(self):
        """Perform exploratory data analysis."""
        print("\n" + "=" * 60)
        print("STEP 2: EXPLORATORY DATA ANALYSIS")
        print("=" * 60)
        
        # Check for missing values
        print("Missing values:")
        print(self.df.isnull().sum())
        
        # Churn distribution
        print(f"\nChurn distribution:")
        churn_counts = self.df['Churn'].value_counts()
        print(churn_counts)
        print(f"Churn rate: {churn_counts[1] / len(self.df) * 100:.2f}%")
        
        # Create visualizations
        self._create_visualizations()
        
    def _create_visualizations(self):
        """Create exploratory visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Churn distribution
        sns.countplot(data=self.df, x='Churn', ax=axes[0,0])
        axes[0,0].set_title('Churn Distribution')
        
        # Tenure vs Churn
        sns.boxplot(data=self.df, x='Churn', y='Tenure', ax=axes[0,1])
        axes[0,1].set_title('Tenure vs Churn')
        
        # Monthly Charges vs Churn
        sns.boxplot(data=self.df, x='Churn', y='MonthlyCharges', ax=axes[1,0])
        axes[1,0].set_title('Monthly Charges vs Churn')
        
        # Contract distribution
        sns.countplot(data=self.df, x='Contract', hue='Churn', ax=axes[1,1])
        axes[1,1].set_title('Contract Type vs Churn')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('data_exploration.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def preprocess_data(self):
        """Preprocess the data for machine learning."""
        print("\n" + "=" * 60)
        print("STEP 3: DATA PREPROCESSING")
        print("=" * 60)
        
        # Create a copy for preprocessing
        df_processed = self.df.copy()
        
        # Remove CustomerID (not useful for prediction)
        if 'CustomerID' in df_processed.columns:
            df_processed = df_processed.drop('CustomerID', axis=1)
            print("Removed CustomerID column")
        
        # Handle missing values
        print(f"Missing values before handling: {df_processed.isnull().sum().sum()}")
        df_processed = df_processed.dropna()
        print(f"Missing values after handling: {df_processed.isnull().sum().sum()}")
        
        # Encode categorical variables
        categorical_columns = df_processed.select_dtypes(include=['object']).columns
        print(f"Categorical columns to encode: {list(categorical_columns)}")
        
        for col in categorical_columns:
            if col != 'Churn':  # Don't encode target variable
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
                self.label_encoders[col] = le
                print(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        # Separate features and target
        self.X = df_processed.drop('Churn', axis=1)
        self.y = df_processed['Churn']
        
        # Store feature names for later use
        self.feature_names = list(self.X.columns)
        
        print(f"Final feature set: {self.feature_names}")
        print(f"Feature matrix shape: {self.X.shape}")
        print(f"Target shape: {self.y.shape}")
        
        return self.X, self.y
    
    def split_data(self, test_size=0.2):
        """Split data into training and testing sets."""
        print("\n" + "=" * 60)
        print("STEP 4: DATA SPLITTING")
        print("=" * 60)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42, stratify=self.y
        )
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Testing set shape: {self.X_test.shape}")
        print(f"Training churn rate: {self.y_train.mean():.3f}")
        print(f"Testing churn rate: {self.y_test.mean():.3f}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def scale_features(self):
        """Scale numerical features."""
        print("\n" + "=" * 60)
        print("STEP 5: FEATURE SCALING")
        print("=" * 60)
        
        # Fit scaler on training data
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("Features scaled successfully")
        print(f"Training data scaled shape: {self.X_train_scaled.shape}")
        print(f"Testing data scaled shape: {self.X_test_scaled.shape}")
        
        return self.X_train_scaled, self.X_test_scaled
    
    def train_models(self):
        """Train multiple models and select the best one."""
        print("\n" + "=" * 60)
        print("STEP 6: MODEL TRAINING")
        print("=" * 60)
        
        # Define models to try
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100)
        }
        
        # Train and evaluate each model
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(self.X_train_scaled, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = model.score(self.X_test_scaled, self.y_test)
            f1 = f1_score(self.y_test, y_pred)
            auc = roc_auc_score(self.y_test, y_pred_proba)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'f1_score': f1,
                'auc': auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        
        # Select best model based on F1 score
        best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
        self.model = results[best_model_name]['model']
        
        print(f"\nBest model: {best_model_name}")
        print(f"Best F1 Score: {results[best_model_name]['f1_score']:.4f}")
        
        self.results = results
        return self.model
    
    def evaluate_model(self):
        """Comprehensive model evaluation."""
        print("\n" + "=" * 60)
        print("STEP 7: MODEL EVALUATION")
        print("=" * 60)
        
        # Get predictions from best model
        y_pred = self.model.predict(self.X_test_scaled)
        y_pred_proba = self.model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Classification report
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"Confusion Matrix:\n{cm}")
        
        # Create evaluation visualizations
        self._create_evaluation_plots(y_pred, y_pred_proba)
        
        # Feature importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            self._plot_feature_importance()
        elif hasattr(self.model, 'coef_'):
            self._plot_coefficients()
    
    def _create_evaluation_plots(self, y_pred, y_pred_proba):
        """Create evaluation visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion matrix heatmap
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_xlabel('Predicted')
        axes[0,0].set_ylabel('Actual')
        
        # ROC curve
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        auc = roc_auc_score(self.y_test, y_pred_proba)
        axes[0,1].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        axes[0,1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0,1].set_xlabel('False Positive Rate')
        axes[0,1].set_ylabel('True Positive Rate')
        axes[0,1].set_title('ROC Curve')
        axes[0,1].legend()
        
        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
        axes[1,0].plot(recall, precision, label='Precision-Recall Curve')
        axes[1,0].set_xlabel('Recall')
        axes[1,0].set_ylabel('Precision')
        axes[1,0].set_title('Precision-Recall Curve')
        axes[1,0].legend()
        
        # Prediction probability distribution
        axes[1,1].hist(y_pred_proba[self.y_test == 0], alpha=0.5, label='No Churn', bins=20)
        axes[1,1].hist(y_pred_proba[self.y_test == 1], alpha=0.5, label='Churn', bins=20)
        axes[1,1].set_xlabel('Prediction Probability')
        axes[1,1].set_ylabel('Count')
        axes[1,1].set_title('Prediction Probability Distribution')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_feature_importance(self):
        """Plot feature importance for tree-based models."""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=True)
            
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(feature_importance_df)), feature_importance_df['importance'])
            plt.yticks(range(len(feature_importance_df)), feature_importance_df['feature'])
            plt.xlabel('Feature Importance')
            plt.title('Feature Importance')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def _plot_coefficients(self):
        """Plot coefficients for linear models."""
        if hasattr(self.model, 'coef_'):
            coef = self.model.coef_[0]
            coef_df = pd.DataFrame({
                'feature': self.feature_names,
                'coefficient': coef
            }).sort_values('coefficient', ascending=True)
            
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(coef_df)), coef_df['coefficient'])
            plt.yticks(range(len(coef_df)), coef_df['feature'])
            plt.xlabel('Coefficient')
            plt.title('Model Coefficients')
            plt.tight_layout()
            plt.savefig('model_coefficients.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def save_model(self, model_path='churn_model.joblib'):
        """Save the trained model and preprocessing objects."""
        print("\n" + "=" * 60)
        print("STEP 8: MODEL SAVING")
        print("=" * 60)
        
        # Create a dictionary with all necessary components
        model_components = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        
        # Save the model
        joblib.dump(model_components, model_path)
        print(f"Model saved to {model_path}")
        
        return model_path
    
    def load_model(self, model_path='churn_model.joblib'):
        """Load a saved model."""
        model_components = joblib.load(model_path)
        self.model = model_components['model']
        self.scaler = model_components['scaler']
        self.label_encoders = model_components['label_encoders']
        self.feature_names = model_components['feature_names']
        print(f"Model loaded from {model_path}")
    
    def predict(self, customer_data):
        """
        Make predictions on new customer data.
        
        Args:
            customer_data (dict): Dictionary containing customer features
            
        Returns:
            dict: Prediction results with probability
        """
        # Convert to DataFrame
        df_new = pd.DataFrame([customer_data])
        
        # Apply preprocessing
        for col in self.label_encoders:
            if col in df_new.columns:
                df_new[col] = self.label_encoders[col].transform(df_new[col])
        
        # Ensure all features are present
        for feature in self.feature_names:
            if feature not in df_new.columns:
                df_new[feature] = 0
        
        # Reorder columns to match training data
        df_new = df_new[self.feature_names]
        
        # Scale features
        df_new_scaled = self.scaler.transform(df_new)
        
        # Make prediction
        prediction = self.model.predict(df_new_scaled)[0]
        probability = self.model.predict_proba(df_new_scaled)[0][1]
        
        return {
            'churn_prediction': bool(prediction),
            'churn_probability': float(probability),
            'retention_probability': float(1 - probability)
        }

def main():
    """Main function to run the complete churn prediction pipeline."""
    print("CUSTOMER CHURN PREDICTION MODEL")
    print("=" * 60)
    print("Telecommunications Company - Machine Learning Solution")
    print("=" * 60)
    
    # Initialize the model
    churn_model = ChurnPredictionModel()
    
    # Step 1: Load data
    df = churn_model.load_data('customer_churn.csv')
    
    # Step 2: Explore data
    churn_model.explore_data()
    
    # Step 3: Preprocess data
    X, y = churn_model.preprocess_data()
    
    # Step 4: Split data
    X_train, X_test, y_train, y_test = churn_model.split_data()
    
    # Step 5: Scale features
    X_train_scaled, X_test_scaled = churn_model.scale_features()
    
    # Step 6: Train models
    model = churn_model.train_models()
    
    # Step 7: Evaluate model
    churn_model.evaluate_model()
    
    # Step 8: Save model
    model_path = churn_model.save_model()
    
    # Step 9: Demonstrate prediction
    print("\n" + "=" * 60)
    print("STEP 9: PREDICTION DEMONSTRATION")
    print("=" * 60)
    
    # Example customer data
    example_customer = {
        'Tenure': 5,
        'MonthlyCharges': 70.0,
        'TotalCharges': 350.0,
        'Contract': 'Month-to-month',
        'PaymentMethod': 'Electronic check'
    }
    
    prediction = churn_model.predict(example_customer)
    print(f"Example customer prediction:")
    print(f"Customer data: {example_customer}")
    print(f"Churn prediction: {prediction['churn_prediction']}")
    print(f"Churn probability: {prediction['churn_probability']:.3f}")
    print(f"Retention probability: {prediction['retention_probability']:.3f}")
    
    print("\n" + "=" * 60)
    print("MODEL DEPLOYMENT READY!")
    print("=" * 60)
    print("The model has been successfully trained and saved.")
    print("It can now be integrated into your CRM system.")
    print(f"Model file: {model_path}")
    print("=" * 60)

if __name__ == "__main__":
    main() 