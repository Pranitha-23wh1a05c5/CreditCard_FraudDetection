import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from agent import fraud_agent

from geopy.distance import geodesic   # ✅ NEW

app = FastAPI(
    title="Credit Card Fraud Detection API",
    version="3.0"
)

model = joblib.load("app/ieee_fraud_model.pkl")

class Transaction(BaseModel):
    TransactionAmt: float
    ProductCD: int
    card1: int
    card2: float
    addr1: float
    addr2: float


@app.get("/")
def home():
    return {"message": "Fraud Detection API running with AI Agent 🚀"}


# 🌍 HOME LOCATION (India)
HOME_COUNTRY = 356
HOME_COORDS = (20.5937, 78.9629)

# 🌍 Country coordinates (ADD MORE later if needed)
country_coords = {
    356: (20.5937, 78.9629),    # India
    840: (37.0902, -95.7129),   # USA
    826: (55.3781, -3.4360),    # UK
    124: (56.1304, -106.3468),  # Canada
    36: (25.2744, 133.7751),    # Australia
    533: (12.5211, -69.9683),   # Aruba ✅ FIXED
    643: (61.5240, 105.3188),   # Russia
    156: (35.8617, 104.1954),   # China
    586: (30.3753, 69.3451),    # Pakistan
    50: (23.6850, 90.3563),     # Bangladesh
    360: (-0.7893, 113.9213),   # Indonesia
    608: (12.8797, 121.7740),   # Philippines
    704: (14.0583, 108.2772),   # Vietnam
    710: (-30.5595, 22.9375),   # South Africa
    404: (-0.0236, 37.9062),    # Kenya
    818: (26.8206, 30.8025),    # Egypt
}


@app.post("/predict")
def predict(data: Transaction):

    try:
        country = int(data.addr2)

        # 🌍 REAL DISTANCE CALCULATION
        coords = country_coords.get(country)

        if coords:
            calculated_distance = int(geodesic(HOME_COORDS, coords).km)
        else:
            calculated_distance = 8000  # fallback (better than fake 3000)

        # 📊 MODEL INPUT
        features = pd.DataFrame([{
            "TransactionAmt": data.TransactionAmt,
            "ProductCD": data.ProductCD,
            "card1": data.card1,
            "card2": data.card2,
            "addr1": data.addr1,
            "addr2": data.addr2,
            "dist1": calculated_distance
        }])

        # 🤖 MODEL PREDICTION
        prob = model.predict_proba(features)[0][1]
        risk_score = prob * 100

        # ⚙ RULE ENGINE
        if data.TransactionAmt > 2000:
            risk_score += 20

        if calculated_distance > 5000:
            risk_score += 15

        if data.addr2 != HOME_COUNTRY:
            risk_score += 10

        very_high_risk_countries = {566,231,694,706,332,140,180}
        high_risk_countries = {643,156,586,50,360,608,704,710,404,818}
        medium_risk_countries = {840,826,124,36,250,276,724,380}

        if country in very_high_risk_countries:
            risk_score += 50
        elif country in high_risk_countries:
            risk_score += 30
        elif country in medium_risk_countries:
            risk_score += 15

        # 🎯 CAP
        risk_score = min(risk_score, 100)

        # 🧠 FINAL LABEL
        if risk_score >= 60:
            prediction = "Fraud"
        elif risk_score >= 35:
            prediction = "Suspicious"
        else:
            prediction = "Safe"

        # 🤖 AI AGENT
        model_result = {
            "prediction": prediction,
            "risk_score": risk_score,
            "distance": calculated_distance
        }

        agent_explanation = fraud_agent(data.dict(), model_result)

        # 📤 RESPONSE
        return {
            "prediction": prediction,
            "fraud_probability": float(round(prob * 100, 2)),  # ✅ NEW
            "fraud_risk_score": float(round(risk_score, 2)),
            "distance_from_home": calculated_distance,
            "agent_analysis": agent_explanation
        }

    except Exception as e:
        return {"error": str(e)}