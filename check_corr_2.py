import pandas as pd
import os

data_path = 'data/environmental_effect_on_egg_production.csv'
if not os.path.exists(data_path):
    data_path = 'Egg_Production.csv'
if not os.path.exists(data_path):
    print('File not found')
else:
    df = pd.read_csv(data_path)
    df['Egg_per_Chicken'] = df['Total_egg_production'] / df['Amount_of_chicken']
    df['Feed_per_Chicken'] = df['Amount_of_Feeding'] / df['Amount_of_chicken']
    feature_cols = [
            'Amount_of_chicken', 'Amount_of_Feeding', 'Ammonia',
            'Temperature', 'Humidity', 'Light_Intensity', 'Noise',
            'Feed_per_Chicken'
        ]
    print(df[feature_cols + ['Egg_per_Chicken']].corr()['Egg_per_Chicken'])
