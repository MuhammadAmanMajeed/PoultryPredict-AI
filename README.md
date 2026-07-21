# 🐔 PoultryPredict AI

**AI-powered egg production forecasting & farm management system for poultry farmers.**

PoultryPredict AI is a full-stack Flask web application that uses a trained machine learning model to predict daily egg production based on flock and environmental conditions — and pairs that prediction with real farm economics, health diagnostics, and intelligent alerts.

🔗 **Live Demo:** [poultry-predict-ai.vercel.app](https://poultry-predict-ai.vercel.app)

---

## ✨ Features

- **📈 Egg Production Prediction** — ML model predicts daily egg output based on flock size, breed, age, season, and environmental readings (temperature, humidity, ammonia, light, noise).
- **💰 Farm Economics Engine** — Calculates feed cost, startup cost, monthly revenue/profit, ROI, and break-even period using realistic Pakistani market rates.
- **🧠 Explainable AI** — Shows feature importance and per-prediction contribution breakdown ("how the AI thought") for full transparency.
- **⚠️ Risk & Health Intelligence** — Flags environmental risk levels, detects anomalies in daily egg logs (possible disease outbreak), and provides a symptom-based disease diagnosis tool.
- **💉 Vaccination Scheduler** — Age-based poultry vaccination schedule.
- **🌾 Feed Optimizer** — Finds the feed amount (g/bird) that maximizes profit under current conditions.
- **📊 Farm Dashboard & Logs** — Daily log tracking (eggs, feed, mortality, bird weight, water, egg weight) with historical charts and mortality trend analysis for early disease warning.
- **🔁 Self-Retraining Model** — Retrains the ML model using accumulated real farm log data for improved accuracy over time.
- **🔐 User Accounts** — Registration/login system with per-user history, alerts, and data reset.
- **📤 CSV Export** — Export farm logs for offline analysis.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask, Flask-CORS |
| Machine Learning | scikit-learn, pandas, NumPy |
| Database | SQLite |
| Frontend | HTML, Jinja2 Templates |
| Deployment | Vercel, Gunicorn |

---

## 📁 Project Structure

```
PoultryPredict-AI/
├── app.py                     # Flask app — routes, API endpoints, prediction logic
├── forecaster.py              # FarmIntelligence — risk, anomaly detection, diagnosis, AI insights
├── generate_large_dataset.py  # Synthetic dataset generator for model training
├── main.py                    # Model training script
├── models/                    # Trained model artifacts (.pkl)
├── static/                    # CSS, JS, assets
├── templates/                 # HTML templates (Jinja2)
├── history.db                 # SQLite database
├── requirements.txt
└── vercel.json                # Vercel deployment config
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/MuhammadAmanMajeed/PoultryPredict-AI.git
cd PoultryPredict-AI

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Train the Model

Run this once to generate the dataset and train the prediction model:

```bash
python main.py
```

### Run the App

```bash
python app.py
```

The app will start on `http://localhost:5050`.

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/predict` | POST | Predict egg production for given flock/environment inputs |
| `/api/batch_predict` | POST | Run predictions for multiple records at once |
| `/api/feed_optimizer` | POST | Find the profit-maximizing feed amount |
| `/api/logs` | POST | Add a daily farm log entry |
| `/api/chart_data` | GET | Historical data for dashboard charts |
| `/api/mortality_trend` | GET | Week-over-week mortality trend analysis |
| `/api/alerts` | GET | Unread farm alerts |
| `/api/health/diagnose` | POST | Symptom-based disease diagnosis |
| `/api/health/vaccinations` | GET | Vaccination schedule by flock age |
| `/api/retrain` | POST | Retrain model using accumulated farm logs |
| `/api/model_stats` | GET | Model performance metrics |
| `/api/export/csv` | GET | Export daily logs as CSV |
| `/api/health` | GET | API health check |

> Most endpoints require an authenticated session (login via `/login`).

---

## 👤 Author

**Muhammad Aman Majeed**


---

## 📄 License

This project is currently unlicensed. Add a `LICENSE` file if you'd like to open it up for reuse.
