from flask import Flask, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import datetime
import os

app = Flask(__name__)

# Change yahan kar sakte ho
TICKER = 'BTC-USD'      # BTC ke liye BTC-USD, Gold ke liye GC=F, stock ke liye AAPL ya RELIANCE.NS
PERIOD = '2y'           # 2 saal ka data – fast aur achha

@app.route('/')
def home():
    result = get_prediction()
    if "error" in result:
        return f"<h1>Error</h1><p>{result['error']}</p>"
    
    html = f"""
    <html>
    <head><title>Prediction - {TICKER}</title>
    <style>
        body {{font-family:Arial; text-align:center; padding:40px; background:#f8f9fa;}}
        h1 {{color:#333;}}
        .up {{color:green; font-size:36px; font-weight:bold;}}
        .down {{color:red; font-size:36px; font-weight:bold;}}
    </style>
    </head>
    <body>
    <h1>{TICKER} Next Day Prediction</h1>
    <p>Date: {result['date']}</p>
    <p>Direction: <span class="{result['direction'].lower()}">{result['direction']}</span></p>
    <p>Confidence: {result['confidence']}</p>
    <p>Last Close: ₹{result['last_price']}</p>
    <p>Backtest Accuracy: {result['accuracy']}</p>
    <p>Reason: {result['reason']}</p>
    <p><small>Refresh karo latest ke liye</small></p>
    </body>
    </html>
    """
    return html

@app.route('/api')
def api():
    return jsonify(get_prediction())

def get_prediction():
    try:
        data = yf.download(TICKER, period=PERIOD, progress=False)
        if data.empty:
            return {"error": "Data nahi mila. Ticker check karo ya internet dekho."}

        data['Return'] = data['Close'].pct_change()
        data['Target'] = np.where(data['Return'].shift(-1) > 0, 1, 0)
        data['MA5'] = data['Close'].rolling(5).mean()
        data['MA20'] = data['Close'].rolling(20).mean()
        data['RSI'] = 100 - (100 / (1 + (data['Close'].diff(1).clip(lower=0).rolling(14).mean() / 
                                  -data['Close'].diff(1).clip(upper=0).rolling(14).mean())))
        data.dropna(inplace=True)

        if len(data) < 30:
            return {"error": "Data bahut kam hai."}

        features = ['MA5', 'MA20', 'RSI']
        X = data[features]
        y = data['Target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        acc = accuracy_score(y_test, model.predict(X_test)) * 100

        latest = data[features].iloc[-1:].values
        pred = model.predict(latest)[0]
        prob = model.predict_proba(latest)[0]
        direction = "UP" if pred == 1 else "DOWN"
        confidence = max(prob) * 100

        importances = model.feature_importances_
        top_idx = np.argmax(importances)
        top_feature = features[top_idx]

        last_price = float(data['Close'].iloc[-1])
        today = datetime.date.today().strftime("%d %b %Y")

        return {
            "date": today,
            "direction": direction,
            "confidence": f"{confidence:.1f}%",
            "last_price": f"{last_price:.2f}",
            "accuracy": f"{acc:.2f}%",
            "reason": f"Strong signal from {top_feature}"
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
