from flask import Flask, jsonify, request
from flask_cors import CORS  # Import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import datetime
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for Frontend connection

# Default setting (Agar kuch na mile)
DEFAULT_PERIOD = '2y'

# --- ROUTE FOR FRONTEND ---
@app.route('/scan', methods=['POST'])
def scan():
    # Frontend se symbol mango
    data = request.json
    ticker = data.get('symbol', 'BTC-USD') # Agar frontend kuch na bheje to BTC lo
    
    result = get_prediction(ticker)
    
    if "error" in result:
        return jsonify({"success": False, "error": result['error']})
    else:
        return jsonify({"success": True, "data": result})

@app.route('/')
def home():
    return "AI Trading Server is Online! 🚀 (Use Frontend to Scan)"

def get_prediction(ticker):
    try:
        # Data download
        data = yf.download(ticker, period=DEFAULT_PERIOD, progress=False)
        
        # Check: Data mila ya nahi?
        if data.empty:
            return {"error": "No Data found. Market Closed or Wrong Symbol."}

        # Calculations
        data['Return'] = data['Close'].pct_change()
        data['Target'] = np.where(data['Return'].shift(-1) > 0, 1, 0)
        data['MA5'] = data['Close'].rolling(5).mean()
        data['MA20'] = data['Close'].rolling(20).mean()
        
        # Simple RSI Calculation
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        data.dropna(inplace=True)

        # Safety: Agar data kam pad gaya
        if len(data) < 30:
            return {"error": "Not enough data for AI analysis."}

        features = ['MA5', 'MA20', 'RSI']
        X = data[features]
        y = data['Target']

        # Training
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Prediction for TOMORROW
        latest = data[features].iloc[-1:].values
        pred = model.predict(latest)[0]
        prob = model.predict_proba(latest)[0]
        
        direction = "UP 🟢" if pred == 1 else "DOWN 🔴"
        confidence = round(max(prob) * 100, 1)
        
        # Color & Logic
        color = "#00e676" if pred == 1 else "#ff1744"
        
        # Agar AI confuse hai (50-55% chance), toh WAIT bolo
        if confidence < 55:
            direction = "WAIT ✋"
            color = "#888"
            reason = "AI Confused. Market Sideways hai."
        else:
            # Find strongest reason
            importances = model.feature_importances_
            top_idx = np.argmax(importances)
            top_feature = features[top_idx]
            rsi_val = round(data['RSI'].iloc[-1], 2)
            reason = f"Strong signal from {top_feature} | RSI: {rsi_val}"

        last_price = float(data['Close'].iloc[-1])

        return {
            "direction": direction,
            "confidence": f"{confidence}%",
            "price": f"{last_price:.2f}",
            "reason": reason,
            "color": color,
            "rsi": round(data['RSI'].iloc[-1], 2)
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
    
