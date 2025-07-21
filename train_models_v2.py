# ============================
# Imports
# ============================
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify

# ============================
# Config
# ============================
random.seed(42)
np.random.seed(42)
account_types = ['ETUDIANT', 'FONCTIONNAIRE', 'PROFESSIONNEL', 'SELF_EMPLOYED', 'ENTREPRISE']
transaction_types = ['VERSEMENT', 'VIREMENT', 'RETRAIT', 'PAIEMENT']
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)

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
    _ = generate_rib(generate_num_compte(), code_banque)
    # Normal amount ranges per account type
    if account_type == 'ETUDIANT': low, high = 100, 3500
    elif account_type == 'FONCTIONNAIRE': low, high = 100, 7000
    elif account_type == 'PROFESSIONNEL': low, high = 100, 20000
    elif account_type == 'SELF_EMPLOYED': low, high = 1, 20000
    elif account_type == 'ENTREPRISE': low, high = 100, 90000
    else: low, high = 100, 10000
    # Inject anomalies outside normal range
    amount = random.uniform(high * 1.5, high * 10) if is_anomaly else random.uniform(low, high)
    return {
        'accountType': account_type,
        'amount': round(amount, 2),
        'createDateTime': random_date().isoformat(),
        'transaction_type': random.choice(transaction_types),
        'is_anomaly': is_anomaly
    }

# ============================
# Data Generation
# ============================
print("⏳ Generating 200,000 transactions...")
transactions = []
for _ in tqdm(range(200_000)):
    atype = random.choice(account_types)
    transactions.append(generate_transaction(atype, is_anomaly=0))

print("🚨 Injecting 12,500 anomalies...")
for atype in account_types:
    for _ in range(2500):
        transactions.append(generate_transaction(atype, is_anomaly=1))

df = pd.DataFrame(transactions)
df.to_csv('transactions_v2.csv', index=False)
print("✅ Dataset saved: transactions_v2.csv")

# ============================
# Feature Engineering
# ============================
def prepare_features(df):
    df = df.copy()

    # Parse ISO 8601 datetimes safely (with or without milliseconds)
    df['createDateTime'] = pd.to_datetime(df['createDateTime'], errors='coerce', format='ISO8601')

    # Drop rows with invalid datetime values
    df = df.dropna(subset=['createDateTime'])

    # Extract datetime-based features
    df['hour'] = df['createDateTime'].dt.hour
    df['day_of_week'] = df['createDateTime'].dt.dayofweek

    # Log transform the amount
    df['amount_log'] = np.log1p(df['amount'])

    return df

# ============================
# Training
# ============================
print("\n🔍 Training Logistic Regression models...")
base_features = ['amount_log', 'hour', 'day_of_week']
categorical = ['transaction_type']
pipeline = Pipeline([
    ('encoder', ColumnTransformer([
        ('onehot_trans', OneHotEncoder(handle_unknown='ignore'), categorical)
    ], remainder='passthrough')),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])

for atype, group in df.groupby('accountType'):
    print(f"Training for {atype} ({len(group)} samples)")
    grp = prepare_features(group)
    X = grp[categorical + base_features]
    y = grp['is_anomaly']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    pipeline.fit(X_train, y_train)

    # Save versioned model pipeline
    v2_path = os.path.join(model_dir, f'v2_model_{atype}.pkl')
    joblib.dump(pipeline, v2_path)
    print(f"✅ Saved versioned model: {v2_path}")

    # Validation metrics
    y_pred = pipeline.predict(X_val)
    print(classification_report(y_val, y_pred, digits=3))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))

# ============================
# Independent Test for 'ETUDIANT'
# ============================
print("\n🧪 Independent test for 'ETUDIANT'...")
test_samples = [
    {'accountType':'ETUDIANT','amount':1500,'createDateTime':datetime.now().isoformat(),
     'transaction_type':'VERSEMENT','is_anomaly':0},
    {'accountType':'ETUDIANT','amount':45000,'createDateTime':datetime.now().isoformat(),
     'transaction_type':'RETRAIT','is_anomaly':1}
]
for _ in range(100): test_samples.append(generate_transaction('ETUDIANT', is_anomaly=0))
for _ in range(20):  test_samples.append(generate_transaction('ETUDIANT', is_anomaly=1))

df_test = pd.DataFrame(test_samples)
df_test = prepare_features(df_test)
X_test = df_test[categorical + base_features]
y_true = df_test['is_anomaly']
model = joblib.load(os.path.join(model_dir, 'v2_model_ETUDIANT.pkl'))
y_pred = model.predict(X_test)

# Metrics
print("\n📊 Test Evaluation Metrics for 'ETUDIANT':")
print(classification_report(y_true, y_pred, digits=3))
print(f"Precision: {precision_score(y_true, y_pred):.3f}")
print(f"Recall:    {recall_score(y_true, y_pred):.3f}")
print(f"F1 Score:  {f1_score(y_true, y_pred):.3f}")
print(f"Accuracy:  {accuracy_score(y_true, y_pred):.3f}")

# Visualization
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d',
            xticklabels=['Normal','Anomaly'], yticklabels=['Normal','Anomaly'])
plt.title("Confusion Matrix - ETUDIANT")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

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
