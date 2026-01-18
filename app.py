# app.py - Fixed version (last_price formatting bug + API route + HTML clean)

from flask import Flask, jsonify, render_template_string
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import datetime
import os

app = Flask(__name__)

TICKER = 'AAPL'  # Change to '^NSEI' or 'RELIANCE.NS' if needed
PERIOD = '2y'    # Light data to avoid memory issues on free tier

def get_data(ticker, period):
    try:
        data = yf.download(ticker, period=period, progress=False)
        if data.empty:
            return None, "No data fetched from Yahoo Finance."
        return data, None
    except Exception as e:
        return None, f"Data error: {str(e)}"

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def add_features(df):
    df['Return'] = df['Close'].pct_change()
    df['Target'] = np.where(df['Return'].shift(-1) > 0, 1, 0)
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['RSI'] = compute_rsi(df['Close'])
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Momentum'] = df['Close'] / df['Close'].shift(1) - 1
    df.dropna(inplace=True)
    return df

def get_prediction():
    data, error = get_data(TICKER, PERIOD)
    if error:
        return {"error": error}, f"<h2>Error</h2><p>{error}</p>"

    data = add_features(data)
    features = ['MA5', 'MA20', 'RSI', 'Volume_Change', 'Momentum']
    X = data[features]
    y = data['Target']

    if len(X) < 20:
        msg = "Not enough data after processing."
        return {"error": msg}, f"<h2>Error</h2><p>{msg}</p>"

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test)) * 100

    latest = data[features].iloc[-1]
    prob = model.predict_proba([latest])[0]
    direction = "UP" if model.predict([latest])[0] == 1 else "DOWN"
    confidence = max(prob) * 100

    importances = model.feature_importances_
    top_idx = np.argmax(importances)
    top_feature = features[top_idx]
    top_val = latest[top_idx]

    last_price = float(data['Close'].iloc[-1])  # Fixed: force float
    today = datetime.date.today().strftime("%d %b %Y")

    html = f"""
    <h2>Prediction for {TICKER} ({today})</h2>
    <p><b>Next Trading Day Direction:</b> <span style="color:{'green' if direction == 'UP' else 'red'}; font-weight:bold; font-size:28px;">{direction}</span></p>
    <p><b>Confidence:</b> {confidence:.1f}%</p>
    <p><b>Reason:</b> Strong signal from {top_feature} = {top_val:.4f}</p>
    <p><b>Last Close Price:</b> ₹{last_price:.2f}</p>
    <p><b>Backtest Accuracy:</b> {acc:.2f}%</p>
    <p><small>(Market closed today? Using last available data)</small></p>
    """

    json_data = {
        "ticker": TICKER,
        "date": today,
        "direction": direction,
        "confidence": f"{confidence:.1f}%",
        "reason": f"Strong signal from {top_feature} = {top_val:.4f}",
        "last_price": f"{last_price:.2f}",
        "accuracy": f"{acc:.2f}%"
    }

    return json_data, html

@app.route('/')
def home():
    _, html = get_prediction()
    full_page = f"""
    <!DOCTYPE html>
    <html>
    <head><title>Trading Prediction</title>
    <style>body {{font-family:Arial; text-align:center; padding:30px; background:#f8f9fa;}}</style>
    </head>
    <body>
    <h1>Aryan's Trading Prediction Tool</h1>
    {html}
    <p>Refresh to update | API: <a href="/api/prediction">/api/prediction (JSON)</a></p>
    </body>
    </html>
    """
    return full_page

@app.route('/api/prediction')
def api_prediction():
    json_data, _ = get_prediction()
    return jsonify(json_data)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
