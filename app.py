from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

app = Flask(__name__)
CORS(app)

# --- CONFIG ---
# Binance API URL (Public & Free)
BINANCE_URL = "https://api.binance.com/api/v3/klines"

# --- DATA FETCHING ENGINE (BINANCE) ---
def get_crypto_data(symbol):
    # Symbol formatting (Frontend se 'BTC-USD' aata hai, Binance ko 'BTCUSDT' chahiye)
    clean_symbol = symbol.replace("-USD", "USDT").replace("-", "").upper()
    
    # Agar Stocks hain (Jaise Reliance), toh filhal error return karenge (Kyunki Binance sirf Crypto deta hai)
    if "USDT" not in clean_symbol and "BTC" not in clean_symbol and "ETH" not in clean_symbol:
        return None, "Stocks data blocked on Render. Try Crypto (BTC, ETH, SOL)."

    try:
        # Binance se 2 saal ka data (Interval 1 Day)
        # Limit 500 candles (Enough for training)
        params = {'symbol': clean_symbol, 'interval': '1d', 'limit': '500'}
        response = requests.get(BINANCE_URL, params=params)
        
        data = response.json()
        
        # Check if Binance gave error
        if isinstance(data, dict) and "code" in data:
            return None, "Invalid Symbol or Binance Error."

        # Convert to Pandas DataFrame
        # Binance format: [Open Time, Open, High, Low, Close, Volume, ...]
        df = pd.DataFrame(data, columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime', 'QAV', 'NoT', 'TBB', 'TBQ', 'Ignore'])
        
        # Hamein sirf Close price chahiye
        df['Close'] = df['Close'].astype(float)
        return df, None

    except Exception as e:
        return None, f"Connection Error: {str(e)}"

# --- AI & MATHS ---
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
    
    df.dropna(inplace=True)
    return df

# --- MAIN ML FUNCTION ---
def get_prediction(symbol):
    # 1. Fetch Data from Binance
    data, error = get_crypto_data(symbol)
    if error: return {"error": error}
    
    try:
        # 2. Prepare Data
        data = add_features(data)
        if len(data) < 30: return {"error": "Not enough data for AI."}

        features = ['MA5', 'MA20', 'RSI']
        X = data[features]
        y = data['Target']

        # 3. Train AI (Random Forest)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # 4. Predict
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
            reason = "Market Confusing. No clear trend."
        else:
            reason = f"AI Confidence: {confidence}% | RSI: {rsi}"

        return {
            "symbol": symbol,
            "price": price,
            "direction": direction,
            "confidence": f"{confidence}%",
            "reason": reason,
            "color": color,
            "rsi": rsi
        }

    except Exception as e:
        return {"error": str(e)}

# --- ROUTES ---
@app.route('/')
def home():
    return "BINANCE AI TRADING SERVER RUNNING 🚀"

@app.route('/scan', methods=['POST'])
def scan():
    data = request.json
    symbol = data.get('symbol', 'BTC-USD')
    result = get_prediction(symbol)
    
    if "error" in result:
        return jsonify({"success": False, "error": result['error']})
    else:
        return jsonify({"success": True, "data": result})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
    
