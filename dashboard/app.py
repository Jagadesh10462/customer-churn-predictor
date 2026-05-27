import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Predictor")
st.markdown("*Powered by XGBoost + Groq AI — Built for Infosys DSE*")
st.divider()

# Sidebar inputs
st.sidebar.header("Customer Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Has Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Has Dependents", ["No", "Yes"])
tenure = st.sidebar.number_input("Tenure (months)", min_value=0, max_value=72, value=12, step=1)
phone = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
payment = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", min_value=0.0, max_value=150.0, value=70.0, step=0.5)
total_charges = st.sidebar.number_input("Total Charges ($)", min_value=0.0, max_value=9000.0, value=1000.0, step=10.0)

# Encode inputs
def encode():
    return {
        "gender": 1 if gender == "Male" else 0,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": 1 if partner == "Yes" else 0,
        "Dependents": 1 if dependents == "Yes" else 0,
        "tenure": tenure,
        "PhoneService": 1 if phone == "Yes" else 0,
        "MultipleLines": ["No phone service", "No", "Yes"].index(multiple_lines),
        "InternetService": ["DSL", "Fiber optic", "No"].index(internet),
        "OnlineSecurity": ["No internet service", "No", "Yes"].index(online_security),
        "OnlineBackup": ["No internet service", "No", "Yes"].index(online_backup),
        "DeviceProtection": ["No internet service", "No", "Yes"].index(device_protection),
        "TechSupport": ["No internet service", "No", "Yes"].index(tech_support),
        "StreamingTV": ["No internet service", "No", "Yes"].index(streaming_tv),
        "StreamingMovies": ["No internet service", "No", "Yes"].index(streaming_movies),
        "Contract": ["Month-to-month", "One year", "Two year"].index(contract),
        "PaperlessBilling": 1 if paperless == "Yes" else 0,
        "PaymentMethod": ["Electronic check", "Mailed check",
                          "Bank transfer (automatic)", "Credit card (automatic)"].index(payment),
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

# Predict button
if st.sidebar.button("🔍 Predict Churn", type="primary"):
    with st.spinner("Analyzing customer..."):
        try:
            response = requests.post(f"{API_URL}/predict", json=encode())
            result = response.json()

            # Risk color
            color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
            risk = result["risk_level"]

            # Top row metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Churn Probability", f"{result['churn_probability']*100:.1f}%")
            col2.metric("Prediction", "Will Churn ❌" if result["churn_prediction"] == 1 else "Will Stay ✅")
            col3.metric("Risk Level", f"{color[risk]} {risk}")

            st.divider()

            # Gauge chart
            col4, col5 = st.columns(2)
            with col4:
                st.subheader("📈 Churn Risk Gauge")
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result["churn_probability"] * 100,
                    title={"text": "Churn Probability %"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "darkred"},
                        "steps": [
                            {"range": [0, 40], "color": "lightgreen"},
                            {"range": [40, 70], "color": "yellow"},
                            {"range": [70, 100], "color": "salmon"}
                        ]
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)

            # SHAP factors
            with col5:
                st.subheader("🔍 Top Churn Factors (SHAP)")
                factors_df = pd.DataFrame(result["top_factors"])
                fig2 = go.Figure(go.Bar(
                    x=factors_df["impact"],
                    y=factors_df["feature"],
                    orientation="h",
                    marker_color="crimson"
                ))
                fig2.update_layout(
                    xaxis_title="Impact Score",
                    yaxis_title="Feature",
                    height=300
                )
                st.plotly_chart(fig2, use_container_width=True)

            st.divider()

            # AI Explanation
            st.subheader("🤖 AI Explanation (Groq Llama)")
            st.info(result["explanation"])

        except Exception as e:
            st.error(f"Error: {str(e)}")

st.divider()

# History section
st.subheader("📋 Recent Predictions")
if st.button("Load History"):
    try:
        history = requests.get(f"{API_URL}/history").json()
        if history:
            df = pd.DataFrame(history)[
                ["id", "churn_probability", "churn_prediction", "created_at"]
            ]
            df["churn_prediction"] = df["churn_prediction"].map(
                {1: "Will Churn ❌", 0: "Will Stay ✅"}
            )
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No predictions yet!")
    except Exception as e:
        st.error(f"Error: {str(e)}")