# Customer Churn Prediction with TensorFlow - VS Code Script

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load Dataset
df = pd.read_csv('customer_churn.csv')
print(df.head())

# Step 3: Preprocessing
# Drop CustomerID
df = df.drop(['CustomerID'], axis=1)

# Encode Categorical Variables
le = LabelEncoder()
df['Contract'] = le.fit_transform(df['Contract'])
df['PaymentMethod'] = le.fit_transform(df['PaymentMethod'])

# Define Features and Target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Scale Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Build and Train TensorFlow Model
model = Sequential([
    Dense(32, activation='relu', input_dim=X_train.shape[1]),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(X_train, y_train, epochs=50, batch_size=2, validation_split=0.2)

# Step 5: Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.4f}')

# Predictions
y_pred = (model.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, y_pred))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Step 6: Save Model
model.save('customer_churn_model.h5')

# TensorFlow Lite Conversion (Optional for Efficiency)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('customer_churn_model.tflite', 'wb') as f:
    f.write(tflite_model)

# Notes:
# - Preprocessing steps (encoding/scaling) must be applied during production prediction
# - The provided dataset is small; for real scenarios, use full production-scale datasets
# - Model is saved and ready for CRM integration

# End of Script
