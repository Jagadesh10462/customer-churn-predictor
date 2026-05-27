from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
import json
from dotenv import load_dotenv
import os
from api.predict import get_prediction

load_dotenv()

app = FastAPI(title="Churn Predictor API", version="1.0")

DB_PATH = "db/churn.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_data TEXT NOT NULL,
            churn_probability REAL NOT NULL,
            churn_prediction INTEGER NOT NULL,
            explanation TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

def get_db():
    return sqlite3.connect(DB_PATH)

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

        db = get_db()
        cursor = db.cursor()
        cursor.execute(
            """INSERT INTO predictions
               (customer_data, churn_probability, churn_prediction, explanation)
               VALUES (?, ?, ?, ?)""",
            (json.dumps(customer.dict()), result["churn_probability"],
             result["churn_prediction"], result["explanation"])
        )
        db.commit()
        db.close()

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
def get_history():
    try:
        db = get_db()
        db.row_factory = sqlite3.Row
        cursor = db.cursor()
        cursor.execute(
            "SELECT * FROM predictions ORDER BY created_at DESC LIMIT 10"
        )
        rows = [dict(row) for row in cursor.fetchall()]
        db.close()
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))