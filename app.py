# app.py

from flask import Flask, jsonify, render_template_string
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

# Config - Light bana diya crash avoid karne ke liye
TICKER = 'AAPL'          # Change kar sakta hai '^NSEI', 'RELIANCE.NS' etc.
PERIOD = '2y'            # 5y se 2y kiya (kam data = fast + no OOM)

def get_data(ticker, period):
    try:
        data = yf.download(ticker, period=period, progress=False)
        if data.empty:
            return None, "No data from yfinance. Check ticker or connection."
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
    df['Return'] = df['Close'].pct_change()
    df['Target'] = np.where(df['Return'].shift(-1) > 0, 1, 0)  # 1=UP, 0=DOWN
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
        return {"error": error}, None, None, None
    
    try:
        data = add_features(data)
        features = ['MA5', 'MA20', 'RSI', 'Volume_Change', 'Momentum']
        X = data[features]
        y = data['Target']
        
        if len(X) < 10:  # Kam data ho toh avoid crash
            return {"error": "Not enough data after features"}, None, None, None
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = RandomForestClassifier(n_estimators=50, random_state=42)  # 100 se 50 kiya light ke liye
        model.fit(X_train, y_train)
        
        acc = accuracy_score(y_test, model.predict(X_test)) if len(y_test) > 0 else 0
        
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
        
        html_result = f"""
        <h2>Prediction for {TICKER} as of {today}</h2>
        <p><b>Next Trading Day:</b> <span style="color: {'green' if direction=='UP' else 'red'}; font-size:28px;"><b>{direction}</b></span></p>
        <p><b>Confidence:</b> {confidence:.1f}%</p>
        <p><b>Reason:</b> Strong signal from {top_feature} ({top_val:.4f})</p>
        <p><b>Last Close:</b> ₹{last_price:.2f}</p>
        <p><b>Backtest Accuracy:</b> {acc*100:.2f}%</p>
        <p><small>Market band hone pe last data use kiya gaya hai</small></p>
        """
        
        json_result = {
            "ticker": TICKER,
            "date": today,
            "direction": direction,
            "confidence": f"{confidence:.1f}%",
            "last_price": f"{last_price:.2f}",
            "reason": f"Strong signal from {top_feature} ({top_val:.4f})",
            "backtest_accuracy": f"{acc*100:.2f}%"
        }
        
        return json_result, html_result, direction, confidence
    
    except Exception as e:
        error_msg = f"Prediction error: {str(e)}\n{traceback.format_exc()}"
        return {"error": error_msg}, None, None, None

@app.route('/')
def home():
    json_data, html, _, _ = get_prediction()
    if "error" in json_data:
        return f"<h1>Error</h1><p>{json_data['error']}</p>"
    
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head><title>Trading Prediction - {TICKER}</title>
    <style>body {{font-family:Arial; text-align:center; padding:30px; background:#f8f9fa;}}</style>
    </head>
    <body>
    <h1>Trading Prediction Tool</h1>
    {html}
    <p>Refresh for update | API: <a href="/api/prediction">/api/prediction</a></p>
    </body>
    </html>
    """
    return render_template_string(full_html)

@app.route('/api/prediction')
def api_prediction():
    json_data, _, _, _ = get_prediction()
    return jsonify(json_data)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
