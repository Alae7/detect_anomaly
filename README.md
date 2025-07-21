# Transaction Anomaly Detection Project

## 🎯 Objectif du projet

Détecter automatiquement les transactions bancaires anormales à l'aide d'un modèle supervisé de **Logistic Regression**, entraîné sur des transactions synthétiques.

## 📝 Contenu

* Génération de 200000 transactions normales et 12500 anomalies
* Feature engineering : `amount_log`, heure, jour de la semaine, `transaction_type`
* Entraînement d'un modèle par type de compte (`ETUDIANT`, `FONCTIONNAIRE`, etc.)
* Évaluation sur un jeu de test indépendant
* API Flask pour prédictions en temps réel

---

## 📁 Structure du projet

```
├── models/                      # Modèles sauvegardés: v2_model_<accountType>.pkl
│   ├── v2_model_ETUDIANT.pkl
│   ├── v2_model_FONCTIONNAIRE.pkl
│   └── ...
├── transactions_v2.csv          # Jeu de données généré (200000 transactions + 12500 anomalies)
├── train_model_v2.py            # Script complet: génération, entraînement, test, AI
└── README.md                    # Documentation du projet
```

---

## 🚀 Exécution

1. **Génération & Entraînement**

   ```bash
   python train_model_v2.py
   ```

   * Génère `transactions_v2.csv`
   * Entraîne et sauvegarde `models/v2_model_<accountType>.pkl`
   * Réalise un test indépendant pour `ETUDIANT`

2. **Lancement de l’API**
   (Le serveur Flask démarre automatiquement à la fin de `train_model_v2.py`)

   ```bash
   python python train_model_v2.py

   ```

---

## 🖥️ API Endpoint

**URL**

```
POST http://localhost:5000/predict
```

**Headers**

| Clé          | Valeur           |
|--------------|------------------|
| Content-Type | application/json |

**Body (JSON)**

| Champ             | Type   | Exemple                 | Description                                    |
|-------------------|--------|-------------------------|------------------------------------------------|
| accountType       | string | `"ETUDIANT"`            | Type de compte (`etudiant`, `fonctionnaire`,…) |
| amount            | number | `45000`                 | Montant brut de la transaction                 |
| createDateTime    | string | `"2025-07-18T14:30:00"` | Date ISO de la transaction                     |
| transaction\_type | string | `"RETRAIT"`             | Type d’opération (`versement`, `virement`,…)   |

**Exemple de requête (curl)**

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "accountType": "ETUDIANT",
        "amount": 45000,
        "createDateTime": "2025-07-18T14:30:00",
        "transaction_type": "RETRAIT"
      }'
```

**Exemple de réponse**

```json
{
  "prediction": "anomaly"
}
```

Pour un montant normal (ex. 1500€) :

```json
{ "prediction": "normal" }
```

---

## 📊 Évaluation

Le jeu de test indépendant pour **`ETUDIANT`** contient :

* 2 cas manuels
* 100 transactions générées normales
* 20 transactions générées anormales

Metrics calculées: précision, rappel, F1-score, accuracy + matrice de confusion.

---

## 📚 Références

* Scikit-learn Logistic Regression: [https://scikit-learn.org/stable/modules/linear\_model.html#logistic-regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
* Flask: [https://flask.palletsprojects.com/](https://flask.palletsprojects.com/)

---

*© 2025 – Détection d’anomalies de transactions*
