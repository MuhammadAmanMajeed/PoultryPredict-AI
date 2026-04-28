import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import os

data_path = 'data/environmental_effect_on_egg_production.csv'
if not os.path.exists(data_path):
    data_path = 'Egg_Production.csv'
df = pd.read_csv(data_path)

df['Egg_per_Chicken'] = df['Total_egg_production'] / df['Amount_of_chicken']
df['Feed_per_Chicken'] = df['Amount_of_Feeding'] / df['Amount_of_chicken']

df['Temp_Stress'] = np.abs(df['Temperature'] - 22) / 10
df['Humidity_Stress'] = np.abs(df['Humidity'] - 60) / 20
df['Ammonia_Stress'] = df['Ammonia'] / 25
df['Environmental_Stress_Index'] = df['Temp_Stress'] + df['Humidity_Stress'] + df['Ammonia_Stress']

feature_cols = [
    'Ammonia', 'Temperature', 'Humidity', 'Light_Intensity', 'Noise', 
    'Feed_per_Chicken', 'Environmental_Stress_Index'
]

X = df[feature_cols]
y = df['Egg_per_Chicken']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("Test R2 Score:", r2_score(y_test, y_pred))

importances = pd.DataFrame({'feature': feature_cols, 'importance': np.round(rf.feature_importances_, 3)})
importances = importances.sort_values('importance', ascending=False)
print("\nFeature Importances:\n", importances)
