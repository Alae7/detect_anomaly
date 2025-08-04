# ============================
# Imports
# ============================
import pandas as pd
import numpy as np
import joblib
import os
from flask import Flask, request, jsonify

# ============================
# Config
# ============================

model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)

# ============================
# Flask API
# ============================
from datetime import datetime
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() or {}
    required = ['accountType','amount','createDateTime','transaction_type']
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({'error':f"Missing fields: {missing}"}),400

    try:
        dt = datetime.fromisoformat(data['createDateTime'])
        hour = dt.hour
        dow = dt.weekday()
    except Exception as e:
        return jsonify({'error':'Invalid createDateTime','details':str(e)}),400

    feat = pd.DataFrame([{
        'transaction_type': data['transaction_type'],
        'amount_log': np.log1p(data['amount']),
        'hour': hour,
        'day_of_week': dow
    }])

    model = joblib.load(os.path.join(model_dir, f"v2_model_{data['accountType']}.pkl"))
    if model is None:
        return jsonify({'error':'Model not found'}),404

    pred = model.predict(feat)[0]
    return jsonify({'prediction':'anomaly' if pred==1 else 'normal'})

if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
