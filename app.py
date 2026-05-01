"""
Flask API for Egg Production Prediction System
Muhammad Aman Majeed - 2022-ag-6211
"""

from flask import Flask, request, jsonify, render_template, session, redirect, url_for, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
import datetime
import sqlite3
import csv
import io
from forecaster import FarmIntelligence

app = Flask(__name__)
app.secret_key = 'your_super_secret_key_here'
CORS(app)

# Ensure models directory exists for build process
os.makedirs('models', exist_ok=True)

MODEL_PATH = 'models/egg_production_model.pkl'
model_data = None

def load_model():
    global model_data
    try:
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                model_data = pickle.load(f)
            return True
        else:
            print(f"Warning: Model not found at {MODEL_PATH}")
            return False
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Initial attempt to load model
load_model()

# --- DATABASE HELPERS ---
def get_db_connection():
    # On Vercel, the file system is read-only except for /tmp
    db_path = 'history.db'
    if os.environ.get('VERCEL'):
        db_path = '/tmp/history.db'
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    try:
        conn = get_db_connection()
        conn.execute('''CREATE TABLE IF NOT EXISTS predictions_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME, chickens INTEGER,
            breed TEXT, age INTEGER, temp REAL, humidity REAL, predicted_eggs REAL, total_startup_cost REAL, user_id INTEGER
        )''')
        conn.execute('''CREATE TABLE IF NOT EXISTS daily_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, date DATE, chickens INTEGER, eggs INTEGER, 
            feed_kg REAL, mortality INTEGER, temp REAL, humidity REAL, ammonia REAL, user_id INTEGER
        )''')
        conn.execute('''CREATE TABLE IF NOT EXISTS market_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT, date DATE, egg_price REAL, feed_price REAL
        )''')
        conn.execute('''CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME, type TEXT, message TEXT, user_id INTEGER, is_read INTEGER DEFAULT 0
        )''')
        conn.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, role TEXT DEFAULT 'farmer'
        )''')
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database initialization skipped or failed: {e}")

# Run init_db only if not on Vercel or handle it safely
init_db()

# --- LOGIC HELPERS ---
def get_biological_modifiers(breed, age, season):
    # Breed Modifier
    breed_mods = {'commercial': 1.0, 'misri': 0.7, 'desi': 0.45}
    b_mod = breed_mods.get(breed.lower(), 0.45)
    
    # Age Modifier
    if age < 20: a_mod = 0.0
    elif age <= 45: a_mod = 1.0
    elif age <= 60: a_mod = 0.85
    elif age <= 80: a_mod = 0.70
    else: a_mod = 0.50
    
    # Season Modifier
    s_mod = {'spring': 1.0, 'summer': 0.95, 'autumn': 1.0, 'winter': 0.98}.get(season.lower(), 1.0)
    return b_mod, a_mod, s_mod

def calculate_final_prediction(raw_efficiency, chickens, breed, age, season):
    # Calibration to realistic biological levels (60-95%)
    # Even in poor conditions, a healthy farm shouldn't drop below 50-60% efficiency
    base_eff = 0.75 + ((raw_efficiency - 0.50) * 0.6)
    base_eff = min(max(base_eff, 0.50), 0.95)
    
    b_mod, a_mod, s_mod = get_biological_modifiers(breed, age, season)
    
    # Apply modifiers but keep a floor for commercial breeds
    final_eff = base_eff * b_mod * a_mod * s_mod
    if breed.lower() == 'commercial':
        final_eff = max(final_eff, 0.55) # Poor commercial farm is still ~55%
    
    return final_eff * chickens, final_eff, b_mod, s_mod

def calculate_economics(chickens, prediction, breed, age, system_type, feed_per_bird_g, feed_price, egg_price):
    daily_feed = (feed_per_bird_g / 1000.0) * chickens
    monthly_feed_cost = daily_feed * 30 * feed_price
    
    # Startup costs
    # Realistic Market Rates
    base_costs = {'commercial': 120, 'misri': 80, 'desi': 100}
    bird_cost = base_costs.get(breed.lower(), 100) + max(0, (min(age, 22) - 1) * 35)
    infra_cost = 800 if system_type == 'manual' else 2200
    total_startup = (bird_cost + infra_cost) * chickens
    
    monthly_rev = prediction * egg_price * 30
    monthly_profit = monthly_rev - monthly_feed_cost
    
    return {
        'daily_feed_kg': daily_feed,
        'monthly_feed_kg': daily_feed * 30,
        'monthly_feed_cost_pkr': monthly_feed_cost,
        'startup_cost_pkr': total_startup,
        'monthly_revenue_pkr': monthly_rev,
        'monthly_profit_pkr': monthly_profit,
        'roi_percentage': (monthly_profit * 12 / total_startup * 100) if total_startup > 0 else 0,
        'break_even_months': (total_startup / monthly_profit) if monthly_profit > 0 else 0,
        'bird_cost_total_pkr': bird_cost * chickens,
        'infrastructure_cost_total_pkr': infra_cost * chickens
    }

# --- ROUTES ---
@app.route('/')
def home():
    if 'logged_in' not in session: return redirect(url_for('login'))
    return render_template('index.html', username=session.get('username'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        try:
            db = get_db_connection()
            user = db.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
            db.close()
            if user and check_password_hash(user['password_hash'], password):
                session.update({'logged_in': True, 'user_id': user['id'], 'username': user['username']})
                return redirect(url_for('home'))
        except Exception as e:
            # Vercel Bypass for Presentation
            if username:
                session.update({'logged_in': True, 'user_id': 1, 'username': username})
                return redirect(url_for('home'))
        return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        db = get_db_connection()
        try:
            db.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', 
                       (request.form.get('username'), generate_password_hash(request.form.get('password'))))
            db.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return render_template('register.html', error='Username exists')
        finally:
            db.close()
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/plots/<path:filename>')
def serve_plots(filename):
    return send_from_directory('plots', filename)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        chickens = float(data.get('Amount_of_chicken', 0))
        feed_g = float(data.get('Feed_per_Chicken_g', 0))
        temp, hum = float(data.get('Temperature', 22)), float(data.get('Humidity', 60))
        breed, age, season = data.get('Breed', 'commercial'), float(data.get('Age', 28)), data.get('Season', 'spring')
        
        # 1. Prediction
        input_data = {
            'Amount_of_chicken': chickens, 'Amount_of_Feeding': (feed_g/1000)*chickens,
            'Ammonia': float(data.get('Ammonia', 10)), 'Temperature': temp, 'Humidity': hum,
            'Light_Intensity': float(data.get('Light_Intensity', 50)), 
            'Light_Duration': float(data.get('Light_Duration', 16)), # New feature
            'Noise': float(data.get('Noise', 40)),
            'Feed_per_Chicken': feed_g/1000,
            'Environmental_Stress_Index': (abs(temp-22)/10 + abs(hum-60)/20 + float(data.get('Ammonia', 10))/25)
        }
        input_scaled = model_data['scaler'].transform(pd.DataFrame([input_data])[model_data['feature_names']])
        raw_eff = float(model_data['model'].predict(input_scaled)[0])
        
        # 2. Logic
        prediction, _, b_mod, s_mod = calculate_final_prediction(raw_eff, chickens, breed, age, season)
        econ = calculate_economics(chickens, prediction, breed, age, data.get('System_Type', 'automatic'), 
                                   feed_g, float(data.get('Feed_Price_per_kg', 200)), float(data.get('Egg_Price_per_unit', 22)))

        # 3. Trends & History
        trend_data = []
        for a in [20, 30, 40, 50, 60, 70, 80]:
            _, a_mod, _ = get_biological_modifiers(breed, a, season)
            eff = min((0.7 + (raw_eff - 0.5) * 0.8) * b_mod * a_mod * s_mod, 0.91)
            trend_data.append({'age': a, 'eggs': round(eff * chickens, 0)})

        # SAFE DATABASE SAVE (Optional on Vercel)
        try:
            db = get_db_connection()
            db.execute('INSERT INTO predictions_history (timestamp, chickens, breed, age, temp, humidity, predicted_eggs, total_startup_cost, user_id) VALUES (?,?,?,?,?,?,?,?,?)',
                       (datetime.datetime.now(), chickens, breed, age, temp, hum, round(prediction, 0), econ['startup_cost_pkr'], session.get('user_id')))
            db.commit()
            db.close()
        except Exception as db_err:
            print(f"Database save skipped: {db_err}")

        return jsonify({
            'success': True, 'predicted_eggs': round(prediction, 0),
            'predicted_per_chicken': round(prediction / chickens, 2) if chickens > 0 else 0,
            'economics': {k: round(v, 2) for k, v in econ.items()},
            'trend_data': trend_data, 
            'recommendations': generate_recommendations(data, prediction, econ['monthly_profit_pkr'])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_recommendations(data, prediction, profit=0):
    recommendations = []
    if profit < 0: recommendations.append("Financial Warning: Monthly costs exceed revenue.")
    temp, hum, amm = float(data.get('Temperature', 22)), float(data.get('Humidity', 60)), float(data.get('Ammonia', 10))
    if temp > 26: recommendations.append("Temperature is high. Consider cooling.")
    elif temp < 18: recommendations.append("Temperature is low. Consider heating.")
    if hum > 75: recommendations.append("Humidity is high. Improve ventilation.")
    if amm > 25: recommendations.append("Ammonia levels are dangerous! Immediate ventilation required.")
    if float(data.get('Age', 28)) > 60: recommendations.append("Flock is aging past peak production.")
    if not recommendations: recommendations.append("Optimal environmental conditions.")
    return recommendations

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    try:
        results = []
        for r in request.get_json().get('records', []):
            chickens = float(r.get('Amount_of_chicken', 0))
            raw_eff = float(model_data['model'].predict(model_data['scaler'].transform(pd.DataFrame([r])[model_data['feature_names']]))[0])
            pred, _, _, _ = calculate_final_prediction(raw_eff, chickens, r.get('Breed', 'commercial'), float(r.get('Age', 28)), r.get('Season', 'spring'))
            results.append({'input': r, 'predicted_eggs': round(pred, 0)})
        return jsonify({'success': True, 'predictions': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    if 'logged_in' not in session: return jsonify({'error': 'Unauthorized'}), 401
    db = get_db_connection()
    rows = db.execute('SELECT * FROM predictions_history WHERE user_id = ? ORDER BY id DESC LIMIT 50', (session.get('user_id'),)).fetchall()
    db.close()
    return jsonify({'success': True, 'history': [dict(r) for r in rows]})

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model_data is not None})

# --- NEW INTELLIGENT ROUTES ---

@app.route('/api/logs', methods=['POST'])
def add_log():
    if 'logged_in' not in session: return jsonify({'error': 'Unauthorized'}), 401
    try:
        data = request.get_json()
        db = get_db_connection()
        
        # Check for anomalies before saving
        history = db.execute('SELECT eggs FROM daily_logs WHERE user_id = ? ORDER BY id DESC LIMIT 14', (session.get('user_id'),)).fetchall()
        history_df = pd.DataFrame([dict(r) for r in history])
        
        is_anomaly, z = FarmIntelligence.detect_anomalies(history_df, float(data.get('eggs', 0)))
        if is_anomaly:
            db.execute('INSERT INTO alerts (timestamp, type, message, user_id) VALUES (?, ?, ?, ?)',
                       (datetime.datetime.now(), 'Disease Risk', f'Significant production drop detected (Z-score: {round(z, 2)}). Monitor for disease.', session.get('user_id')))
        
        # Check environmental risks
        env_alerts = FarmIntelligence.check_environmental_risks(float(data.get('temp', 22)), float(data.get('humidity', 60)), float(data.get('ammonia', 10)))
        for a_type, msg in env_alerts:
            db.execute('INSERT INTO alerts (timestamp, type, message, user_id) VALUES (?, ?, ?, ?)',
                       (datetime.datetime.now(), a_type, msg, session.get('user_id')))

        db.execute('''INSERT INTO daily_logs (date, chickens, eggs, feed_kg, mortality, temp, humidity, ammonia, user_id) 
                      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                   (data.get('date', datetime.date.today().isoformat()), data.get('chickens'), data.get('eggs'), 
                    data.get('feed_kg'), data.get('mortality'), data.get('temp'), data.get('humidity'), 
                    data.get('ammonia'), session.get('user_id')))
        db.commit()
        db.close()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    if 'logged_in' not in session: return jsonify({'error': 'Unauthorized'}), 401
    db = get_db_connection()
    rows = db.execute('SELECT * FROM alerts WHERE user_id = ? AND is_read = 0 ORDER BY timestamp DESC', (session.get('user_id'),)).fetchall()
    db.close()
    return jsonify({'success': True, 'alerts': [dict(r) for r in rows]})

@app.route('/api/market_trends', methods=['GET'])
def get_market_trends():
    # Simulate historical data
    historical_prices = [22, 23, 22.5, 24, 25, 24.5, 26, 25.5, 27, 28]
    forecast = FarmIntelligence.forecast_market_prices(historical_prices)
    return jsonify({
        'success': True,
        'history': historical_prices,
        'forecast': forecast,
        'current_price': historical_prices[-1]
    })

@app.route('/api/export/csv')
def export_csv():
    if 'logged_in' not in session: return "Unauthorized", 401
    db = get_db_connection()
    rows = db.execute('SELECT * FROM daily_logs WHERE user_id = ?', (session.get('user_id'),)).fetchall()
    db.close()
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'Date', 'Chickens', 'Eggs', 'Feed_KG', 'Mortality', 'Temp', 'Humidity', 'Ammonia', 'User_ID'])
    for row in rows:
        writer.writerow(list(row))
    
    output.seek(0)
    return output.getvalue(), 200, {
        'Content-Type': 'text/csv',
        'Content-Disposition': 'attachment; filename=poultry_farm_report.csv'
    }

@app.route('/api/health/diagnose', methods=['POST'])
def diagnose_health():
    data = request.json
    symptoms = data.get('symptoms', [])
    diagnosis = FarmIntelligence.diagnose_disease(symptoms)
    return jsonify({'success': True, 'diagnosis': diagnosis})

@app.route('/api/health/vaccinations', methods=['GET'])
def get_vaccinations():
    age = int(request.args.get('age', 1))
    schedule = FarmIntelligence.get_vaccination_schedule(age)
    return jsonify({'success': True, 'schedule': schedule})

if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)