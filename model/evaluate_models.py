# model/evaluate_models.py

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# Load test data
X_test, y_test = joblib.load("trained_models/test_data.pkl")

# Load scaler
scaler = joblib.load("trained_models/scaler.pkl")

models = {
    "Logistic Regression": joblib.load("trained_models/logistic_model.pkl"),
    "Decision Tree": joblib.load("trained_models/decision_tree_model.pkl"),
    "KNN": joblib.load("trained_models/knn_model.pkl"),
    "Naive Bayes": joblib.load("trained_models/naive_bayes_model.pkl"),
    "Random Forest": joblib.load("trained_models/random_forest_model.pkl"),
    "XGBoost": joblib.load("trained_models/xgboost_model.pkl")
}

results = []

for name, model in models.items():

    if name in ["Logistic Regression", "KNN"]:
        X_eval = scaler.transform(X_test)
    else:
        X_eval = X_test

    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)[:, 1]

    results.append([
        name,
        accuracy_score(y_test, y_pred),
        roc_auc_score(y_test, y_prob),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred),
        matthews_corrcoef(y_test, y_pred)
    ])

results_df = pd.DataFrame(results, columns=[
    "Model",
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1",
    "MCC"
])

print(results_df.sort_values(by="F1", ascending=False))
