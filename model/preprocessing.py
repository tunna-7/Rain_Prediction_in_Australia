# model/preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):

    # Make explicit copy first
    df = df.copy()

    df = df.dropna(subset=["RainTomorrow"]).copy()

    df["RainTomorrow"] = df["RainTomorrow"].map({"Yes": 1, "No": 0})
    df = df.dropna(subset=["RainTomorrow"])
    df["RainTomorrow"] = df["RainTomorrow"].astype(int)

    df["RainToday"] = df["RainToday"].map({"Yes": 1, "No": 0})
    df["RainToday"] = df["RainToday"].fillna(0).astype(int)

    y = df["RainTomorrow"].astype(int)
    X = df.drop("RainTomorrow", axis=1)
    
    # Convert Date to datetime
    X["Date"] = pd.to_datetime(X["Date"])

    # Extract useful features
    X["Year"] = X["Date"].dt.year
    X["Month"] = X["Date"].dt.month
    X["Day"] = X["Date"].dt.day

    # Drop original Date column
    X = X.drop("Date", axis=1)

    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(include="object").columns.tolist()

    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    X[num_cols] = num_imputer.fit_transform(X[num_cols])
    X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    feature_columns = X.columns.tolist()

    return X, y, encoders, num_imputer, cat_imputer, feature_columns

