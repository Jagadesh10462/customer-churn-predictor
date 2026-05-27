import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

def train():
    # Load processed data
    df = pd.read_csv("data/processed_churn.csv")
    print(f"✅ Loaded processed data: {df.shape}")

    # Split features and target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    # XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=5174/1869,  # handle class imbalance
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)
    print("✅ XGBoost model trained!")

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n📊 Model Performance:")
    print(f"   Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"   F1 Score  : {f1_score(y_test, y_pred):.4f}")
    print(f"   ROC-AUC   : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")

    # Save model
    joblib.dump(model, "model/churn_model.pkl")
    print("✅ Model saved → model/churn_model.pkl")

    # SHAP values
    print("\n⚡ Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Top 5 important features
    shap_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": np.abs(shap_values).mean(axis=0)
    }).sort_values("importance", ascending=False)

    print("\n🔥 Top 5 Features driving churn:")
    print(shap_importance.head(5).to_string(index=False))

    joblib.dump(explainer, "model/shap_explainer.pkl")
    print("\n✅ SHAP explainer saved → model/shap_explainer.pkl")

if __name__ == "__main__":
    train()