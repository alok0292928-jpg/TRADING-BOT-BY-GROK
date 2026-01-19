from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

app = Flask(__name__)
CORS(app)

# --- CONFIG ---
TRAINING_PERIOD = '2y' 

# --- DATA FETCHING ---
def get_data(ticker):
    try:
        data = yf.download(ticker, period=TRAINING_PERIOD, progress=False)
        if data.empty:
            return None, "No Data Found."
        return data, None
    except Exception as e:
        return None, str(e)

# --- INDICATORS ---
def add_features(df):
    df = df.copy()
    df['Return'] = df['Close'].pct_change()
    df['Target'] = np.where(df['Return'].shift(-1) > 0, 1, 0)
    
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Momentum'] = df['Close'] / df['Close'].shift(1) - 1
    
    df.dropna(inplace=True)
    return df

# --- ML ENGINE ---
def get_ml_prediction(ticker):
    data, error = get_data(ticker)
    if error: return {"error": error}
    
    try:
        data = add_features(data)
        if len(data) < 30: return {"error": "Not enough data."}

        features = ['MA5', 'MA20', 'RSI', 'Momentum']
        X = data[features]
        y = data['Target']

        # Train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
        model.fit(X_train, y_train)

        # Predict
        latest = data[features].iloc[-1:].values
        pred = model.predict(latest)[0]
        prob = model.predict_proba(latest)[0]
        
        confidence = round(max(prob) * 100, 1)
        rsi = round(data['RSI'].iloc[-1], 2)
        price = round(data['Close'].iloc[-1], 2)

        direction = "UP 🟢" if pred == 1 else "DOWN 🔴"
        color = "#00e676" if pred == 1 else "#ff1744"
        
        if confidence < 55:
            direction = "WAIT ✋"
            color = "#888"
            reason = "Market Confusing. No clear signal."
        else:
            reason = f"AI Signal Strong ({confidence}%). RSI: {rsi}"

        return {
            "symbol": ticker,
            "price": price,
            "direction": direction,
            "confidence": f"{confidence}%",
            "reason": reason,
            "color": color,
            "rsi": rsi
        }
    except Exception as e:
        return {"error": str(e)}

@app.route('/')
def home():
    return "AI Trading Brain Active 🧠"

@app.route('/scan', methods=['POST'])
def scan():
    data = request.json
    symbol = data.get('symbol', 'BTC-USD')
    result = get_ml_prediction(symbol)
    if "error" in result:
        return jsonify({"success": False, "error": result['error']})
    return jsonify({"success": True, "data": result})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
    
