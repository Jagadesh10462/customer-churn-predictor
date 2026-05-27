from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mysql.connector
import json
from dotenv import load_dotenv
import os
from api.predict import get_prediction

load_dotenv()

app = FastAPI(title="Churn Predictor API", version="1.0")

def get_db():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME")
    )

class CustomerData(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def root():
    return {"message": "Churn Predictor API is running ✅"}

@app.post("/predict")
def predict(customer: CustomerData):
    try:
        result = get_prediction(customer.dict())

        # Save to MySQL
        db = get_db()
        cursor = db.cursor()
        cursor.execute(
            """INSERT INTO predictions 
               (customer_data, churn_probability, churn_prediction, explanation) 
               VALUES (%s, %s, %s, %s)""",
            (json.dumps(customer.dict()), result["churn_probability"],
             result["churn_prediction"], result["explanation"])
        )
        db.commit()
        cursor.close()
        db.close()

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
def get_history():
    try:
        db = get_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute(
            "SELECT * FROM predictions ORDER BY created_at DESC LIMIT 10"
        )
        rows = cursor.fetchall()
        cursor.close()
        db.close()
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))