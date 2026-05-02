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
        Weighted symptom-based diagnostic engine.
        symptoms: list of strings (e.g., ['green_diarrhea', 'coughing'])
        """
        knowledge_base = {
            'Newcastle Disease (ND)': {
                'signs': {'green_diarrhea': 2.5, 'nervous_signs': 2.0, 'coughing': 1.0, 'gasping': 1.0},
                'severity': 'Critical',
                'treatment': 'No cure for ND. Support with vitamins and electrolytes. Vaccinate healthy birds immediately.',
                'meds': ['Vitamin AD3E', 'Electrolytes', 'Immune Boosters']
            },
            'Infectious Bronchitis (IB)': {
                'signs': {'misshapen_eggs': 2.0, 'sneezing': 1.5, 'wet_eyes': 1.0, 'coughing': 1.0},
                'severity': 'High',
                'treatment': 'Antibiotics to prevent secondary infections. Increase room temperature.',
                'meds': ['Tylosin', 'Doxycycline', 'Menthol spray']
            },
            'Coccidiosis': {
                'signs': {'bloody_diarrhea': 3.0, 'huddled_birds': 1.5, 'weight_loss': 1.0},
                'severity': 'High',
                'treatment': 'Anticoccidial drugs in water. Keep litter dry.',
                'meds': ['Amprolium', 'Toltrazuril', 'Sulfa drugs']
            },
            'Coryza': {
                'signs': {'swollen_face': 2.0, 'foul_smell': 2.0, 'nasal_discharge': 1.5, 'wet_eyes': 1.0},
                'severity': 'Medium',
                'treatment': 'Strict biosecurity. Antibiotic treatment.',
                'meds': ['Oxytetracycline', 'Erythromycin']
            },
            'Fowl Cholera': {
                'signs': {'swollen_face': 1.5, 'green_diarrhea': 1.5, 'weight_loss': 1.0},
                'severity': 'High',
                'treatment': 'Antibiotics in water or feed. Eliminate rodents.',
                'meds': ['Sulfonamides', 'Tetracycline']
            }
        }
        
        results = []
        user_symptoms_count = len(symptoms)
        if user_symptoms_count == 0: return []
        
        for disease, info in knowledge_base.items():
            disease_symptoms = info['signs']
            matched_symptoms = [s for s in symptoms if s in disease_symptoms]
            
            if matched_symptoms:
                matched_weight = sum(disease_symptoms[s] for s in matched_symptoms)
                total_disease_weight = sum(disease_symptoms.values())
                
                # Weight contribution:
                # 1. How much of the disease's "core identity" is covered (recall)
                recall = matched_weight / total_disease_weight
                # 2. How many of the user's symptoms fit this disease (precision)
                precision = len(matched_symptoms) / user_symptoms_count
                
                score = ((precision * 0.4) + (recall * 0.6)) * 100
                
                results.append({
                    'disease': disease,
                    'severity': info['severity'],
                    'score': round(score, 1),
                    'treatment': info['treatment'],
                    'meds': info['meds'],
                    'matched_signs': len(matched_symptoms)
                })
        
        # Sort by score descending, then alphabetically by disease name (Deterministic Lock)
        return sorted(results, key=lambda x: (-x['score'], x['disease']))

    @staticmethod
    def get_vaccination_schedule(age_weeks):
        # Comprehensive Commercial Schedule including Adult Boosters
        schedule = [
            {'week': 0, 'vaccine': "Marek's Disease (HVT)", 'disease': "Marek's"},
            {'week': 1, 'vaccine': 'ND + IB (Live Spray/Ocular)', 'disease': 'Newcastle & Bronchitis'},
            {'week': 2, 'vaccine': 'IBD / Gumboro (Live Drinking Water)', 'disease': 'Infectious Bursal Disease'},
            {'week': 3, 'vaccine': 'ND LaSota (Booster)', 'disease': 'Newcastle'},
            {'week': 4, 'vaccine': 'Avian Influenza (H9N2 Killed)', 'disease': 'Bird Flu'},
            {'week': 5, 'vaccine': 'IBD Booster', 'disease': 'Gumboro'},
            {'week': 8, 'vaccine': 'Fowl Pox (Wing Web)', 'disease': 'Pox'},
            {'week': 10, 'vaccine': 'Infectious Coryza (Killed)', 'disease': 'Coryza'},
            {'week': 12, 'vaccine': 'Avian Encephalomyelitis (AE)', 'disease': 'AE'},
            {'week': 16, 'vaccine': 'ND+IB+EDS (Multi-strain Killed)', 'disease': 'Multi-protection'},
            {'week': 24, 'vaccine': 'Deworming (Water Medication)', 'disease': 'Internal Parasites'},
            {'week': 28, 'vaccine': 'ND + IB (Killed Booster)', 'disease': 'Newcastle & Bronchitis'},
            {'week': 40, 'vaccine': 'ND + IB (Killed Booster)', 'disease': 'Newcastle & Bronchitis'}
        ]
        return [v for v in schedule if v['week'] >= age_weeks]
