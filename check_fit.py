import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load model and scaler
with open('models/egg_production_model.pkl', 'rb') as f:
    data = pickle.load(f)

model = data['model']
scaler = data['scaler']
features = data['feature_names']

# Load data
df = pd.read_csv("Egg_Production_Large.csv")

if 'Active_Birds' in df.columns:
    df.rename(columns={
        'Active_Birds': 'Amount_of_chicken',
        'Amount_of_Feeding_kg': 'Amount_of_Feeding',
        'Ammonia_ppm': 'Ammonia',
        'Temperature_C': 'Temperature',
        'Humidity_pct': 'Humidity',
        'Light_Intensity_lux': 'Light_Intensity',
        'Eggs_Produced': 'Total_egg_production'
    }, inplace=True)
elif 'active_birds' in df.columns:
    df.rename(columns={'active_birds': 'Amount_of_chicken'}, inplace=True)

np.random.seed(42)
if 'Light_Duration' not in df.columns:
    df['Light_Duration'] = np.random.uniform(12, 17, len(df))
if 'Noise' not in df.columns:
    df['Noise'] = np.random.uniform(30, 60, len(df))

df['Total_egg_production'] = np.clip(df['Total_egg_production'], 0, df['Amount_of_chicken'])
df['Feed_per_Chicken'] = df['Amount_of_Feeding'] / df['Amount_of_chicken']
df['Egg_per_Chicken'] = df['Total_egg_production'] / df['Amount_of_chicken']
df['Environmental_Stress_Index'] = (abs(df['Temperature'] - 22) / 10 + abs(df['Humidity'] - 60) / 20 + df['Ammonia'] / 25)

X = df[features]
y = df['Egg_per_Chicken']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

train_pred = model.predict(X_train_scaled)
test_pred = model.predict(X_test_scaled)

train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)

print(f"Model Type: {type(model).__name__}")
print(f"Train R2: {train_r2:.4f}")
print(f"Test R2: {test_r2:.4f}")
