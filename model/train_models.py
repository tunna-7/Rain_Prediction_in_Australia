# model/train_models.py

import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from preprocessing import load_data, preprocess_data

# Load data
df = load_data("dataset/weatherAUS.csv")
X, y, encoders, num_imputer, cat_imputer, feature_columns = preprocess_data(df)

# print("y dtype:", y.dtype)
# print("Unique values in y:", y.unique())

# Save preprocessing objects
joblib.dump(encoders, "trained_models/encoders.pkl")
joblib.dump(num_imputer, "trained_models/num_imputer.pkl")
joblib.dump(cat_imputer, "trained_models/cat_imputer.pkl")
joblib.dump(feature_columns, "trained_models/feature_columns.pkl")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

joblib.dump((X_test, y_test), "trained_models/test_data.pkl")

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, "trained_models/scaler.pkl")

models = {
    "logistic": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(),
    "knn": KNeighborsClassifier(),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(),
    "xgboost": XGBClassifier(eval_metric="logloss")
}

for name, model in models.items():

    if name in ["logistic", "knn"]:
        model.fit(X_train_scaled, y_train)
    else:
        model.fit(X_train, y_train)

    joblib.dump(model, f"trained_models/{name}_model.pkl")

print("Training complete.")
