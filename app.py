from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import datetime
import os
import traceback

app = Flask(__name__)
CORS(app)  # Yeh zaroori hai taaki Frontend connect ho sake

# Config - Default Settings
DEFAULT_PERIOD = '2y' 

def get_data(ticker, period):
    try:
        # Progress false taaki logs na bhare
        data = yf.download(ticker, period=period, progress=False)
        if data.empty:
            return None, "No data found. Market might be closed or invalid symbol."
        return data, None
    except Exception as e:
        return None, f"Data fetch error: {str(e)}"

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def add_features(df):
    df = df.copy()
    df['Return'] = df['Close'].pct_change()
    df['Target'] = np.where(df['Return'].shift(-1) > 0, 1, 0)  # 1=UP, 0=DOWN
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['RSI'] = compute_rsi(df['Close'])
    df['Momentum'] = df['Close'] / df['Close'].shift(1) - 1
    df.dropna(inplace=True)
    return df

def get_ml_prediction(ticker):
    data, error = get_data(ticker, DEFAULT_PERIOD)
    if error:
        return {"error": error}
    
    try:
        data = add_features(data)
        features = ['MA5', 'MA20', 'RSI', 'Momentum'] # Volume hata diya safety ke liye
        
        # Safety: Agar data bohot kam bacha ho
        if len(data) < 20:
            return {"error": "Not enough data points for ML"}
        
        X = data[features]
        y = data['Target']
        
        # Train Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Model Training (Lightweight)
        model = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Latest Prediction
        latest = data[features].iloc[-1:].values
        prob = model.predict_proba(latest)[0]
        prediction = model.predict(latest)[0]
        
        direction = "UP 🟢" if prediction == 1 else "DOWN 🔴"
        confidence = round(max(prob) * 100, 1)
        
        # Top Feature Reason
        rsi_val = round(data['RSI'].iloc[-1], 2)
        last_price = round(data['Close'].iloc[-1], 2)
        
        # Logic to Color
        color = "#00e676" if prediction == 1 else "#ff1744"
        if confidence < 55:
            direction = "WAIT ✋"
            color = "#888"
            reason = f"AI Confused ({confidence}%). Market Sideways."
        else:
            reason = f"AI Confidence: {confidence}% | RSI: {rsi_val}"

        return {
            "symbol": ticker,
            "price": last_price,
            "direction": direction,
            "confidence": f"{confidence}%",
            "reason": reason,
            "color": color,
            "rsi": rsi_val
        }
    
    except Exception as e:
        return {"error": f"ML Logic Error: {str(e)}"}

# --- ROUTES ---

@app.route('/')
def home():
    return "AI Trading Brain is Running 🧠 (Use Frontend to Interact)"

# Yeh Route Frontend use karega
@app.route('/scan', methods=['POST'])
def scan_endpoint():
    data = request.json
    symbol = data.get('symbol', 'BTC-USD') # Default BTC
    
    result = get_ml_prediction(symbol)
    
    if "error" in result:
        return jsonify({"success": False, "error": result['error']})
    else:
        return jsonify({"success": True, "data": result})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
        
