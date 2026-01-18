# app.py   ← Yeh naam exact rakhna Render ke liye

from flask import Flask, render_template_string
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import datetime
import os

app = Flask(__name__)

# Config
TICKER = 'AAPL'   # Change kar sakta hai jaise 'RELIANCE.NS' ya '^NSEI'
PERIOD = '5y'

def get_data(ticker, period):
    data = yf.download(ticker, period=period, progress=False)
    if data.empty:
        return None, "No data fetched! Check ticker/internet."
    return data, None

def add_features(df):
    df['Return'] = df['Close'].pct_change()
    df['Target'] = np.where(df['Return'].shift(-1) > 0, 1, 0)  # 1=UP, 0=DOWN
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['RSI'] = compute_rsi(df['Close'])
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Momentum'] = df['Close'] / df['Close'].shift(1) - 1
    df.dropna(inplace=True)
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_prediction():
    data, error = get_data(TICKER, PERIOD)
    if error:
        return error, None, None, None
    
    data = add_features(data)
    features = ['MA5', 'MA20', 'RSI', 'Volume_Change', 'Momentum']
    X = data[features]
    y = data['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    
    latest = data[features].iloc[-1:].values
    prob = model.predict_proba(latest)[0]
    direction = "UP" if model.predict(latest)[0] == 1 else "DOWN"
    confidence = max(prob) * 100
    
    importances = model.feature_importances_
    top_idx = np.argmax(importances)
    top_feature = features[top_idx]
    top_val = latest[0][top_idx]
    
    last_price = data['Close'].iloc[-1]
    today = datetime.date.today().strftime("%d %b %Y")
    
    result = f"""
    <h2>Prediction as of {today}</h2>
    <p><b>Direction for next trading day:</b> <span style="color: {'green' if direction=='UP' else 'red'}; font-size:24px;"><b>{direction}</b></span></p>
    <p><b>Confidence:</b> {confidence:.1f}%</p>
    <p><b>Reason:</b> Strong signal from {top_feature} = {top_val:.4f}</p>
    <p><b>Last Close Price:</b> ₹{last_price:.2f}</p>
    <p><b>Backtest Accuracy:</b> {acc*100:.2f}%</p>
    """
    return None, result, direction, confidence

@app.route('/')
def home():
    error, result, _, _ = get_prediction()
    if error:
        return f"<h1>Error</h1><p>{error}</p>"
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head><title>Trading Prediction - {TICKER}</title>
    <style>body {{font-family: Arial; text-align:center; padding:20px;}}</style>
    </head>
    <body>
    <h1>Trading Prediction Tool</h1>
    {result}
    <p>Refresh page for latest prediction (market closed pe bhi chalega last data se)</p>
    </body>
    </html>
    """
    return render_template_string(html)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
