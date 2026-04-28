"""
Train ML Models for PoultryPredict
Muhammad Aman Majeed - 2022-ag-6211
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
import xgboost as xgb

print("="*60)
print("TRAINING POULTRYPREDICT MODELS")
print("="*60)

# Load dataset
try:
    df = pd.read_csv('data/poultry_data.csv')
    print(f"✓ Dataset loaded: {df.shape}")
except:
    print("✗ Dataset not found at data/poultry_data.csv")
    exit()

# Feature engineering
df['Feed_per_Chicken'] = df['Amount_of_Feeding'] / df['Amount_of_chicken']
df['Stress_Index'] = (
    abs(df['Temperature'] - 22) / 10 +
    abs(df['Humidity'] - 60) / 20 +
    df['Ammonia'] / 50
)

# Prepare features
features = ['Amount_of_chicken', 'Amount_of_Feeding', 'Temperature',
            'Humidity', 'Ammonia', 'Light_Intensity', 'Noise', 'Feed_per_Chicken']
X = df[features]
y = df['Total_egg_production']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"✓ Training set: {X_train.shape}")
print(f"✓ Test set: {X_test.shape}")

# Train XGBoost model
print("\nTraining XGBoost model...")
egg_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
egg_model.fit(X_train, y_train)

# Evaluate
y_pred = egg_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"✓ Model R² Score: {r2:.4f}")

# Save model
import os
os.makedirs('models', exist_ok=True)

model_data = {
    'model': egg_model,
    'feature_names': features,
    'r2_score': r2
}

with open('models/egg_predictor.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print(f"\n✓ Model saved to: models/egg_predictor.pkl")
print("="*60)
print("TRAINING COMPLETE!")
print("="*60)
print("You can now run: python app.py")