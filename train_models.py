# ========== Imports ==========
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify
import threading

# ========== Config ==========
random.seed(42)
numpy_rng = np.random.default_rng(42)
account_types = ['etudiant', 'fonctionnaire', 'professionnel', 'self_employed', 'entreprise']
transaction_types = ['versement', 'virement', 'retrait', 'paiement']
os.makedirs("models", exist_ok=True)

# ========== Helper Functions ==========
def random_date():
    start = datetime.now() - timedelta(days=90)
    end = datetime.now()
    return start + (end - start) * random.random()

def generate_num_compte():
    return random.randint(1_000_000_000, 9_999_999_999)

def generate_rib(num_compte: int, code_banque: str) -> dict:
    base = f"{code_banque}{num_compte:010}00"
    cle = 97 - (int(base) % 97)
    rib = f"{code_banque}{num_compte:010}{cle:02}"
    return {"rib": rib, "cle": cle}

def generate_transaction(account_type, is_anomaly=0):
    code_banque = random.choice(["01345", "01234", "00456"])
    rib_source = generate_rib(generate_num_compte(), code_banque)
    # Define normal ranges based on account  type
    if account_type == 'etudiant':
        low, high = 100, 3500
    elif account_type == 'fonctionnaire':
        low, high = 100, 7000
    elif account_type == 'professionnel':
        low, high = 100, 20000
    elif account_type == 'self_employed':
        low, high = 1, 20000
    elif account_type == 'entreprise':
        low, high = 100, 90000
    else:
        low, high = 100, 10000

    # If anomaly, use a higher range
    if is_anomaly:
        amount = random.uniform(high * 1.5, high * 10)
    else:
        amount = random.uniform(low, high)

    return {
        'amount': round(amount, 2),
        'type': random.choice(transaction_types),
        'rib': int(rib_source["rib"]),
        'createDateTime': random_date(),
        'accountType': account_type,
        'is_anomaly': is_anomaly
    }

# ========== Data Generation ==========
print("⏳ Generating 200,000 transactions...")
transactions = []
for _ in tqdm(range(200000)):
    acc_type = random.choice(account_types)
    transactions.append(generate_transaction(acc_type))

# Inject anomalies evenly
print("🚨 Injecting anomalies...")
for acc_type in account_types:
    for _ in range(2500):
        transactions.append(generate_transaction(acc_type, is_anomaly=1))

df = pd.DataFrame(transactions)
df.to_csv("transactions.csv", index=False)
print("✅ Dataset saved: transactions.csv")

# ========== Train Logistic Regression Models ==========
print("\n🔍 Training models for each account type...")
for acc_type, group in df.groupby("accountType"):
    print(f"Training {acc_type} (n={len(group)})")
    group['amount_log'] = np.log1p(group['amount'])
    group['hour'] = pd.to_datetime(group['createDateTime']).dt.hour
    group['day_of_week'] = pd.to_datetime(group['createDateTime']).dt.dayofweek

    X = group[['amount_log', 'hour', 'day_of_week']]
    y = group['is_anomaly']

    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    model.fit(X, y)
    joblib.dump(model, f"models/logreg_model_{acc_type}.pkl")
    print(f"Saved: models/logreg_model_{acc_type}.pkl")

# ========== Independent Test for 'fonctionnaire' ==========
print("\n🧪 Independent test for 'fonctionnaire'...")
# Create manual test data or generate new
test_data = []
# Example manual points
test_data.extend([
    {'amount': 3500, 'createDateTime': datetime.now(), 'accountType': 'fonctionnaire', 'is_anomaly': 0},
    {'amount': 48000, 'createDateTime': datetime.now(), 'accountType': 'fonctionnaire', 'is_anomaly': 1}
])
# Or synthetic
for _ in range(100): test_data.append(generate_transaction('fonctionnaire', is_anomaly=0))
for _ in range(20): test_data.append(generate_transaction('fonctionnaire', is_anomaly=1))

df_test = pd.DataFrame(test_data)
# Feature engineering
df_test['amount_log'] = np.log1p(df_test['amount'])
# assume createDateTime datetime dtype
df_test['hour'] = pd.to_datetime(df_test['createDateTime']).dt.hour
df_test['day_of_week'] = pd.to_datetime(df_test['createDateTime']).dt.dayofweek

X_test = df_test[['amount_log', 'hour', 'day_of_week']]
y_true = df_test['is_anomaly']
# Load test model
model_test = joblib.load("models/logreg_model_fonctionnaire.pkl")
y_pred = model_test.predict(X_test)

# Metrics
print(classification_report(y_true, y_pred, digits=3))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

# Visualization
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d',
            xticklabels=['Normal','Anomaly'], yticklabels=['Normal','Anomaly'], cmap='Blues')
plt.title("Confusion Matrix - fonctionnaire")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ========== Flask API ==========
app = Flask(__name__)
models_cache = {}
def load_model(acc_type):
    if acc_type not in models_cache:
        path = f"models/logreg_model_{acc_type}.pkl"
        models_cache[acc_type] = joblib.load(path) if os.path.exists(path) else None
    return models_cache[acc_type]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() or {}
    missing = [f for f in ['accountType','amount','hour','day_of_week'] if f not in data]
    if missing:
        return jsonify({'error':f"Missing: {missing}"}),400
    amt, hr, dow = data['amount'], data['hour'], data['day_of_week']
    model = load_model(data['accountType'])
    if not model: return jsonify({'error':'Model not found'}),404
    df_feat = pd.DataFrame([{'amount_log':np.log1p(amt),'hour':hr,'day_of_week':dow}])
    pred = model.predict(df_feat)[0]
    return jsonify({'prediction':'anomaly' if pred==1 else 'normal'})

if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000,debug=True)
