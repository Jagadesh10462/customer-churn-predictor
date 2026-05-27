import joblib
import numpy as np
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

model = joblib.load("model/churn_model.pkl")
explainer = joblib.load("model/shap_explainer.pkl")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

FEATURE_NAMES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]

def get_prediction(customer: dict):
    features = np.array([[customer[f] for f in FEATURE_NAMES]])

    prob = model.predict_proba(features)[0][1]
    prediction = int(prob >= 0.5)

    shap_values = explainer.shap_values(features)[0]
    top_factors = sorted(
        zip(FEATURE_NAMES, shap_values),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:3]

    factors_text = "\n".join([
        f"- {name}: impact score {value:.3f}"
        for name, value in top_factors
    ])

    prompt = f"""A telecom customer has a {prob*100:.1f}% probability of churning.

Top 3 factors driving this prediction:
{factors_text}

In 2-3 simple sentences, explain to a non-technical business manager:
1. Will this customer likely churn?
2. What are the main reasons?
3. What action should be taken?

Be direct and practical."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )
    explanation = response.choices[0].message.content

    return {
        "churn_probability": round(float(prob), 4),
        "churn_prediction": prediction,
        "risk_level": "High" if prob >= 0.7 else "Medium" if prob >= 0.4 else "Low",
        "explanation": explanation,
        "top_factors": [{"feature": n, "impact": round(float(v), 4)}
                        for n, v in top_factors]
    }