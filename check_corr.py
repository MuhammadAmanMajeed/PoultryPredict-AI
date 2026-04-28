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
    print(df[['Amount_of_chicken', 'Amount_of_Feeding', 'Feed_per_Chicken', 'Egg_per_Chicken', 'Total_egg_production']].corr())
