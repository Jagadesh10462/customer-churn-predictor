import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

def preprocess_data():
    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    
    print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Drop customerID (not useful for prediction)
    df.drop(columns=["customerID"], inplace=True)

    # Fix TotalCharges — it has spaces instead of nulls
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Convert target column — Yes=1, No=0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Encode all categorical columns
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=["str"]).columns

    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
        print(f"   Encoded: {col}")

    # Save processed file
    df.to_csv("data/processed_churn.csv", index=False)
    print(f"\n✅ Processed data saved → data/processed_churn.csv")
    print(f"   Churn distribution:\n{df['Churn'].value_counts()}")

    return df

if __name__ == "__main__":
    preprocess_data()