import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(page_title="Rain Prediction Dashboard", layout="wide")
st.title("🌦️ Rain in Australia - ML Dashboard")
st.write("Interactive comparison of 6 ML models with live prediction.")

# =====================================================
# Load All Saved Artifacts
# =====================================================

@st.cache_resource
def load_artifacts():

    models = {
        "Logistic Regression": joblib.load("trained_models/logistic_model.pkl"),
        "Decision Tree": joblib.load("trained_models/decision_tree_model.pkl"),
        "KNN": joblib.load("trained_models/knn_model.pkl"),
        "Naive Bayes": joblib.load("trained_models/naive_bayes_model.pkl"),
        "Random Forest": joblib.load("trained_models/random_forest_model.pkl"),
        "XGBoost": joblib.load("trained_models/xgboost_model.pkl"),
    }

    scaler = joblib.load("trained_models/scaler.pkl")
    encoders = joblib.load("trained_models/encoders.pkl")
    num_imputer = joblib.load("trained_models/num_imputer.pkl")
    cat_imputer = joblib.load("trained_models/cat_imputer.pkl")
    feature_columns = joblib.load("trained_models/feature_columns.pkl")
    X_test, y_test = joblib.load("trained_models/test_data.pkl")

    return models, scaler, encoders, num_imputer, cat_imputer, feature_columns, X_test, y_test


models, scaler, encoders, num_imputer, cat_imputer, feature_columns, X_test, y_test = load_artifacts()

# =====================================================
# Sidebar Model Selection
# =====================================================

st.sidebar.header("🔽 Model Selection")
selected_model_name = st.sidebar.selectbox("Choose Model", list(models.keys()))
selected_model = models[selected_model_name]

# =====================================================
# Show Metrics for Selected Model
# =====================================================

st.subheader(f"📊 Performance Metrics - {selected_model_name}")

if selected_model_name in ["Logistic Regression", "KNN"]:
    X_eval = scaler.transform(X_test)
else:
    X_eval = X_test

y_pred = selected_model.predict(X_eval)
y_prob = selected_model.predict_proba(X_eval)[:, 1]

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "AUC": roc_auc_score(y_test, y_prob),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1 Score": f1_score(y_test, y_pred),
    "MCC": matthews_corrcoef(y_test, y_pred)
}

metric_df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
st.dataframe(metric_df)
st.bar_chart(metric_df.set_index("Metric"))

# =====================================================
# Prediction Section
# =====================================================

st.markdown("---")
st.subheader("🎯 Make a Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    MinTemp = st.number_input("MinTemp", value=10.0)
    MaxTemp = st.number_input("MaxTemp", value=20.0)
    Rainfall = st.number_input("Rainfall", value=0.0)
    WindSpeed9am = st.number_input("WindSpeed9am", value=10.0)

with col2:
    WindSpeed3pm = st.number_input("WindSpeed3pm", value=15.0)
    Humidity9am = st.slider("Humidity9am", 0, 100, 50)
    Humidity3pm = st.slider("Humidity3pm", 0, 100, 50)
    Pressure9am = st.number_input("Pressure9am", value=1010.0)

with col3:
    Pressure3pm = st.number_input("Pressure3pm", value=1010.0)
    Temp9am = st.number_input("Temp9am", value=15.0)
    Temp3pm = st.number_input("Temp3pm", value=18.0)
    RainToday = st.selectbox("Rain Today?", ["No", "Yes"])

RainToday = 1 if RainToday == "Yes" else 0

# =====================================================
# Default Features (10)
# =====================================================

# Building Date
today = datetime.today()
year = today.year
month = today.month
day = today.day


default_values = {
    "Year": year,
    "Month": month,
    "Day": day,
    "Location": "Sydney",
    "Evaporation": 5.0,
    "Sunshine": 7.0,
    "WindGustDir": "N",
    "WindGustSpeed": 30.0,
    "WindDir9am": "N",
    "WindDir3pm": "N",
    "Cloud9am": 5.0,
    "Cloud3pm": 5.0,
}

# =====================================================
# Feature Transparency Section
# =====================================================

st.markdown("---")
st.subheader("📋 Feature Information")

st.markdown("### ✅ User Input Features (12)")
st.write([
    "MinTemp", "MaxTemp", "Rainfall", "WindSpeed9am", "WindSpeed3pm",
    "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm",
    "Temp9am", "Temp3pm", "RainToday"
])

st.markdown("### ⚙️ Default Features Used (10)")

default_df = pd.DataFrame(default_values.items(), columns=["Feature", "Default Value"])
default_df["Default Value"] = default_df["Default Value"].astype(str)
st.table(default_df)

# =====================================================
# Combine All 22 Features
# =====================================================

input_dict = {
    "MinTemp": MinTemp,
    "MaxTemp": MaxTemp,
    "Rainfall": Rainfall,
    "WindSpeed9am": WindSpeed9am,
    "WindSpeed3pm": WindSpeed3pm,
    "Humidity9am": Humidity9am,
    "Humidity3pm": Humidity3pm,
    "Pressure9am": Pressure9am,
    "Pressure3pm": Pressure3pm,
    "Temp9am": Temp9am,
    "Temp3pm": Temp3pm,
    "RainToday": RainToday,
}

input_dict.update(default_values)
input_df = pd.DataFrame([input_dict])

# =====================================================
# Apply Preprocessing
# =====================================================

# Ensure exact training column order BEFORE any transform
input_df = input_df[feature_columns]

# Split columns exactly like training
num_cols = num_imputer.feature_names_in_
cat_cols = cat_imputer.feature_names_in_

# Apply imputers using training column order
input_df[num_cols] = num_imputer.transform(input_df[num_cols])
input_df[cat_cols] = cat_imputer.transform(input_df[cat_cols])

for col in cat_cols:
    if col in encoders:
        input_df[col] = encoders[col].transform(input_df[col])

input_df = input_df[feature_columns]

# =====================================================
# Predict
# =====================================================

if st.button("Predict"):

    if selected_model_name in ["Logistic Regression", "KNN"]:
        input_processed = scaler.transform(input_df)
    else:
        input_processed = input_df

    prediction = selected_model.predict(input_processed)[0]
    probability = selected_model.predict_proba(input_processed)[0][1]

    if prediction == 1:
        st.error(f"🌧️ It WILL rain tomorrow (Probability: {probability:.2f})")
    else:
        st.success(f"☀️ It will NOT rain tomorrow (Probability: {probability:.2f})")
