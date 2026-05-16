"""
Cleaned Original ML Pipeline
Egg Production Prediction System
Muhammad Aman Majeed
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import xgboost as xgb

warnings.filterwarnings("ignore")


class EggProductionPredictor:

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None

        self.features = [
            'Ammonia',
            'Temperature',
            'Humidity',
            'Light_Intensity',
            'Light_Duration',
            'Noise',
            'Feed_per_Chicken',
            'Environmental_Stress_Index'
        ]

    def load_data(self):
        if self.file_path.endswith(".xlsx"):
            self.df = pd.read_excel(self.file_path)
        else:
            self.df = pd.read_csv(self.file_path)

        if 'Active_Birds' in self.df.columns:
            self.df.rename(columns={
                'Active_Birds': 'Amount_of_chicken',
                'Amount_of_Feeding_kg': 'Amount_of_Feeding',
                'Ammonia_ppm': 'Ammonia',
                'Temperature_C': 'Temperature',
                'Humidity_pct': 'Humidity',
                'Light_Intensity_lux': 'Light_Intensity',
                'Eggs_Produced': 'Total_egg_production'
            }, inplace=True)
        elif 'active_birds' in self.df.columns:
            self.df.rename(columns={'active_birds': 'Amount_of_chicken'}, inplace=True)

        print("Dataset Loaded Successfully")

    def feature_engineering(self):
        np.random.seed(42)
        
        # Synthesize missing features if they don't exist
        if 'Light_Duration' not in self.df.columns:
            # Typical poultry lighting is 14-16 hours
            self.df['Light_Duration'] = np.random.uniform(12, 17, len(self.df))
        
        if 'Noise' not in self.df.columns:
            self.df['Noise'] = np.random.uniform(30, 60, len(self.df))

        noise = np.random.normal(0, 0.02, len(self.df))
        self.df['Total_egg_production'] = self.df['Total_egg_production'] * (1 + noise)
        self.df['Total_egg_production'] = np.clip(self.df['Total_egg_production'], 0, self.df['Amount_of_chicken'])
        self.df['Feed_per_Chicken'] = self.df['Amount_of_Feeding'] / self.df['Amount_of_chicken']
        self.df['Egg_per_Chicken'] = self.df['Total_egg_production'] / self.df['Amount_of_chicken']
        self.df['Environmental_Stress_Index'] = (abs(self.df['Temperature'] - 22) / 10 + abs(self.df['Humidity'] - 60) / 20 + self.df['Ammonia'] / 25)
        print("Feature Engineering Completed")

    def prepare_data(self):
        X = self.df[self.features]
        y = self.df['Egg_per_Chicken']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_models(self, X_train, X_test, y_train, y_test):
        model_list = {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "XGBoost": xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
            "SVR": SVR(C=100, gamma=0.1)
        }
        best_score = -999
        for name, model in model_list.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            score = r2_score(y_test, pred)
            self.models[name] = model
            if score > best_score:
                best_score = score
                self.best_model = model
        print(f"Best Model Selected: {type(self.best_model).__name__}")

    def hyperparameter_tuning(self, X_train, y_train):
        if isinstance(self.best_model, RandomForestRegressor):
            params = {"n_estimators": [100, 200], "max_depth": [10, 20, None]}
            grid = GridSearchCV(RandomForestRegressor(random_state=42), params, cv=5, scoring="r2")
            grid.fit(X_train, y_train)
            self.best_model = grid.best_estimator_
        elif isinstance(self.best_model, xgb.XGBRegressor):
            params = {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]}
            grid = GridSearchCV(xgb.XGBRegressor(random_state=42), params, cv=5, scoring="r2")
            grid.fit(X_train, y_train)
            self.best_model = grid.best_estimator_
        print("Hyperparameter Tuning Completed")

    def run_professional_validation(self, X_train, X_test, y_train, y_test):
        os.makedirs("static/plots", exist_ok=True)
        sns.set_theme(style="darkgrid")

        # 1. Learning Curves
        train_sizes, train_scores, test_scores = learning_curve(
            self.best_model, X_train, y_train, cv=5, scoring='r2', 
            train_sizes=np.linspace(0.1, 1.0, 10)
        )
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color="#fbbf24", label='Training Score')
        plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color="#3b82f6", label='Test Score')
        plt.title('Learning Curves (Overfitting Detection)')
        plt.xlabel('Training Samples')
        plt.ylabel('R2 Score')
        plt.legend()
        plt.savefig('static/plots/learning_curves.png', transparent=True)
        plt.close()

        # 2. Residual Plot
        y_pred = self.best_model.predict(X_test)
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        plt.title('Residual Plot (Error Analysis)')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.savefig('static/plots/residual_plot.png', transparent=True)
        plt.close()

        # 3. Feature Importance
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(indices)), importances[indices], color="#fbbf24", align='center')
            plt.yticks(range(len(indices)), [self.features[i] for i in indices])
            plt.title('Model Feature Importance')
            plt.savefig('static/plots/feature_importance_model.png', transparent=True)
            plt.close()

        # 4. Noise Robustness Simulation
        noise_levels = np.linspace(0, 0.2, 5)
        r2_scores = []
        for n in noise_levels:
            X_noisy = X_test + np.random.normal(0, n, X_test.shape)
            r2_scores.append(r2_score(y_test, self.best_model.predict(X_noisy)))
        plt.figure(figsize=(10, 6))
        plt.plot(noise_levels, r2_scores, 'o-', color="#10b981", linewidth=2)
        plt.title('Real-World Noise Robustness Simulation')
        plt.xlabel('Input Noise Level')
        plt.ylabel('Model Reliability (R2)')
        plt.savefig('static/plots/noise_simulation.png', transparent=True)
        plt.close()

        # Print Final Evaluation Metrics
        print("\n--- FINAL MODEL EVALUATION METRICS ---")
        print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.4f}")
        print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred):.4f}")
        print(f"R-Squared (R2) Score:     {r2_score(y_test, y_pred):.4f}")
        print("--------------------------------------\n")

        print("Professional Validation Suite Completed")

    def save_model(self):
        os.makedirs("models", exist_ok=True)
        X_train, X_test, y_train, y_test = self.prepare_data()
        y_pred = self.best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        metrics = {
            'r2': round(r2_score(y_test, y_pred), 4),
            'mae': round(mean_absolute_error(y_test, y_pred), 4),
            'mse': round(mse, 4),
            'rmse': round(mse ** 0.5, 4),
            'dataset_size': len(self.df)
        }
        data = {
            "model": self.best_model,
            "scaler": self.scaler,
            "feature_names": self.features,
            "metrics": metrics
        }
        with open("models/egg_production_model.pkl", "wb") as f:
            pickle.dump(data, f)
        print(f"Model Saved Successfully | R2: {metrics['r2']} | MAE: {metrics['mae']}")

    def predict(self, input_dict):
        df = pd.DataFrame([input_dict])[self.features]
        scaled = self.scaler.transform(df)
        return self.best_model.predict(scaled)[0]


def main():
    predictor = EggProductionPredictor("Egg_Production_Large.csv")
    predictor.load_data()
    predictor.feature_engineering()
    X_train, X_test, y_train, y_test = predictor.prepare_data()
    predictor.train_models(X_train, X_test, y_train, y_test)
    predictor.hyperparameter_tuning(X_train, y_train)
    predictor.run_professional_validation(X_train, X_test, y_train, y_test)
    predictor.save_model()

if __name__ == "__main__":
    main()