import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class FarmIntelligence:
    @staticmethod
    def calculate_fcr(feed_kg, eggs_count):
        """
        Calculate Feed Conversion Ratio (FCR)
        For egg production, FCR is often (Feed consumed / Egg mass) or (Feed consumed / Dozen eggs).
        We'll use (Feed kg / Eggs produced) as a simple indicator.
        """
        if eggs_count == 0: return 0
        return round(feed_kg / eggs_count, 4)

    @staticmethod
    def detect_anomalies(history_df, current_value, threshold=2.0):
        """
        Detect anomalies in production drops using Z-score.
        """
        if len(history_df) < 7: return False, 0 # Not enough data for trend
        
        mean_prod = history_df['eggs'].mean()
        std_prod = history_df['eggs'].std()
        
        if std_prod == 0: return False, 0
        
        z_score = (current_value - mean_prod) / std_prod
        
        # If z-score is negative and below threshold, it's a significant drop
        if z_score < -threshold:
            return True, z_score
        return False, z_score

    @staticmethod
    def forecast_market_prices(historical_prices, days=30):
        """
        Simulate market price forecasting using linear trend + seasonality.
        In a real app, this would use Prophet or LSTM.
        """
        if len(historical_prices) < 10:
            # Not enough data, return static forecast with noise
            last_price = historical_prices[-1] if len(historical_prices) > 0 else 24
            return [last_price + np.random.uniform(-0.5, 0.5) for _ in range(days)]
        
        # Simple linear projection
        x = np.arange(len(historical_prices))
        y = np.array(historical_prices)
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        
        future_x = np.arange(len(historical_prices), len(historical_prices) + days)
        forecast = p(future_x)
        
        # Add some "seasonality" noise
        forecast = forecast + (np.sin(future_x / 7.0) * 0.5)
        return forecast.tolist()

    @staticmethod
    def check_environmental_risks(temp, humidity, ammonia):
        alerts = []
        if temp > 28: alerts.append(("Environment", "Heat Stress Warning: Temperature exceeds 28°C."))
        if temp < 15: alerts.append(("Environment", "Cold Stress Warning: Temperature below 15°C."))
        if humidity > 75: alerts.append(("Environment", "High Humidity Warning: Risk of respiratory issues."))
        if ammonia > 20: alerts.append(("Environment", "High Ammonia Levels: Immediate ventilation required!"))
        return alerts

    @staticmethod
    def diagnose_disease(symptoms):
        """
        Simple symptom-based diagnostic engine.
        symptoms: list of strings (e.g., ['diarrhea', 'coughing'])
        """
        knowledge_base = {
            'Newcastle Disease (ND)': {
                'signs': ['green_diarrhea', 'nervous_signs', 'coughing', 'gasping'],
                'treatment': 'No cure for ND. Support with vitamins and electrolytes. Vaccinate healthy birds immediately.',
                'meds': ['Vitamin AD3E', 'Electrolytes', 'Immune Boosters']
            },
            'Infectious Bronchitis (IB)': {
                'signs': ['sneezing', 'wet_eyes', 'misshapen_eggs', 'coughing'],
                'treatment': 'Antibiotics to prevent secondary infections. Increase room temperature.',
                'meds': ['Tylosin', 'Doxycycline', 'Menthol spray']
            },
            'Coccidiosis': {
                'signs': ['bloody_diarrhea', 'huddled_birds', 'weight_loss'],
                'treatment': 'Anticoccidial drugs in water. Keep litter dry.',
                'meds': ['Amprolium', 'Toltrazuril', 'Sulfa drugs']
            },
            'Coryza': {
                'signs': ['swollen_face', 'foul_smell', 'nasal_discharge'],
                'treatment': 'Strict biosecurity. Antibiotic treatment.',
                'meds': ['Oxytetracycline', 'Erythromycin']
            }
        }
        
        results = []
        for disease, info in knowledge_base.items():
            matches = len(set(symptoms) & set(info['signs']))
            if matches > 0:
                score = (matches / len(info['signs'])) * 100
                results.append({
                    'disease': disease,
                    'score': round(score, 1),
                    'treatment': info['treatment'],
                    'meds': info['meds']
                })
        
        return sorted(results, key=lambda x: x['score'], reverse=True)

    @staticmethod
    def get_vaccination_schedule(age_weeks):
        schedule = [
            {'week': 1, 'vaccine': 'ND + IB (Spray/Ocular)', 'disease': 'Newcastle & Bronchitis'},
            {'week': 2, 'vaccine': 'Gumboro (Live)', 'disease': 'IBD'},
            {'week': 3, 'vaccine': 'ND LaSota', 'disease': 'Newcastle'},
            {'week': 5, 'vaccine': 'IBD Booster', 'disease': 'Gumboro'},
            {'week': 8, 'vaccine': 'Fowl Pox', 'disease': 'Pox'},
            {'week': 16, 'vaccine': 'ND+IB+EDS (Killed)', 'disease': 'Multi-protection'}
        ]
        return [v for v in schedule if v['week'] >= age_weeks]
