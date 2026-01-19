from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

app = Flask(__name__)
CORS(app)  # Frontend ko connect karne ke liye zaroori hai

# --- AI CONFIG ---
# 2 Saal ka data best hota hai training ke liye (Not too old, not too short)
TRAINING_PERIOD = '2y' 

# --- DATA FETCHING ---
def get_data(ticker):
    try:
        # Aaj Monday hai, data fatak se aayega
        data = yf.download(ticker, period=TRAINING_PERIOD, progress=False)
        
        # Check: Agar ticker galat hai ya data nahi aaya
        if data.empty:
            return None, "Symbol Galat hai ya Market Data Unavailable hai."
        
        return data, None
    except Exception as e:
        return None, f"Error: {str(e)}"

# --- MATHS (INDICATORS) ---
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def add_features(df):
    df = df.copy()
    # Target: Agar kal price upar gaya toh 1, niche gaya toh 0
    df['Return'] = df['Close'].pct_change()
    df['Target'] = np.where(df['Return'].shift(-1) > 0, 1, 0)
    
    # Indicators (AI ke inputs)
    df['MA5'] = df['Close'].rolling(5).mean()    # Short Term Trend
    df['MA20'] = df['Close'].rolling(20).mean()  # Medium Term Trend
    df['RSI'] = compute_rsi(df['Close'])         # Overbought/Oversold
    df['Momentum'] = df['Close'] / df['Close'].shift(1) - 1 # Power check
    
    df.dropna(inplace=True) # Khali rows hatao
    return df

# --- MAIN PREDICTION ENGINE ---
def get_ml_prediction(ticker):
    # 1. Data Lao
    data, error = get_data(ticker)
    if error: return {"error": error}
    
    try:
        # 2. Indicators Lagao
        data = add_features(data)
        
        # Safety: Agar calculation ke baad data kam bacha
        if len(data) < 30:
            return {"error": "Not enough data for AI Training."}

        features = ['MA5', 'MA20', 'RSI', 'Momentum']
        X = data[features]
        y = data['Target']

        # 3. AI ko Train karo (Pichle data par)
        # Hum data split karte hain taaki check kar sakein AI sahi seekh raha hai ya nahi
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        model = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
        model.fit(X_train, y_train)

        # 4. Aaj ka Prediction (Abhi ka haal dekh kar)
        latest_data = data[features].iloc[-1:].values
        current_price = round(data['Close'].iloc[-1], 2)
        
        prediction = model.predict(latest_data)[0]      # 1 (UP) or 0 (DOWN)
        probabilities = model.predict_proba(latest_data)[0] # Confidence %
        
        confidence = round(max(probabilities) * 100, 1)
        rsi_val = round(data['RSI'].iloc[-1], 2)

        # 5. Result Formatter
        if prediction == 1:
            direction = "UP 🟢"
            color = "#00e676" # Neon Green
        else:
            direction = "DOWN 🔴"
            color = "#ff1744" # Neon Red
            
        # Logic Reason
        if confidence < 55:
            # Agar AI confuse hai (50-55% chance), toh risk mat lo
            direction = "WAIT ✋"
            color = "#888"
            reason = f"Market Confusing hai (Confidence: {confidence}%). Trade mat lo."
        else:
            reason = f"AI Signal Strong hai ({confidence}%). RSI: {rsi_val}"

        return {
            "symbol": ticker,
            "price": current_price,
            "direction": direction,
            "confidence": f"{confidence}%",
            "reason": reason,
            "color": color,
            "rsi": rsi_val
        }

    except Exception as e:
        return {"error": f"AI Engine Fail: {str(e)}"}

# --- ROUTES ---

@app.route('/')
def home():
    return "🔥 AI TRADING SERVER IS LIVE (Happy Monday!) 🔥"

@app.route('/scan', methods=['POST'])
def scan():
    # Frontend se symbol pakdo
    req_data = request.json
    symbol = req_data.get('symbol', 'BTC-USD')
    
    # AI ko kaam pe lagao
    result = get_ml_prediction(symbol)
    
    if "error" in result:
        return jsonify({"success": False, "error": result['error']})
    else:
        return jsonify({"success": True, "data": result})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
        
    
