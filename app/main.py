import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Credit Card Fraud Detection API",
    version="1.0"
)

# Load trained model
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
    return {"message": "Fraud Detection API running"}


# Home country
HOME_COUNTRY = 356  # India

# Approximate distances from India (km)
country_distance = {
    356: 0,
    840: 13000,
    826: 7200,
    124: 12000,
    36: 10000,
    566: 8000,
    231: 8200,
    694: 8200,
    706: 5000,
    643: 4300,
    156: 3800,
    586: 800,
    50: 1400,
    360: 5000,
    608: 5500,
    704: 4000,
    710: 7800,
    404: 4500,
    818: 4400
}


@app.post("/predict")
def predict(data: Transaction):

    try:

        # Automatic distance calculation
        country = int(data.addr2)
        calculated_distance = country_distance.get(country, 3000)

        # Create dataframe
        features = pd.DataFrame([{
            "TransactionAmt": data.TransactionAmt,
            "ProductCD": data.ProductCD,
            "card1": data.card1,
            "card2": data.card2,
            "addr1": data.addr1,
            "addr2": data.addr2,
            "dist1": calculated_distance
        }])

        # ML prediction
        prob = model.predict_proba(features)[0][1]
        risk_score = prob * 100


        # ---------------- RULE ENGINE ---------------- #

        if data.TransactionAmt > 2000:
            risk_score += 20

        if calculated_distance > 5000:
            risk_score += 15

        if data.addr2 != HOME_COUNTRY:
            risk_score += 10


        # -------- GLOBAL FRAUD COUNTRY RISK -------- #

        very_high_risk_countries = {
            566,  # Nigeria
            231,  # Liberia
            694,  # Sierra Leone
            706,  # Somalia
            332,  # Haiti
            140,  # Central African Republic
            180   # Congo
        }

        high_risk_countries = {
            643,  # Russia
            156,  # China
            586,  # Pakistan
            50,   # Bangladesh
            360,  # Indonesia
            608,  # Philippines
            704,  # Vietnam
            710,  # South Africa
            404,  # Kenya
            818   # Egypt
        }

        medium_risk_countries = {
            840,  # USA
            826,  # UK
            124,  # Canada
            36,   # Australia
            250,  # France
            276,  # Germany
            724,  # Spain
            380   # Italy
        }

        country = int(data.addr2)

        if country in very_high_risk_countries:
            risk_score += 50

        elif country in high_risk_countries:
            risk_score += 30

        elif country in medium_risk_countries:
            risk_score += 15


        # Cap score at 100
        risk_score = min(risk_score, 100)


        # -------- FRAUD CLASSIFICATION -------- #

        if risk_score >= 60:
            prediction = "Fraud"

        elif risk_score >= 35:
            prediction = "Suspicious"

        else:
            prediction = "Safe"


        return {
            "prediction": prediction,
            "fraud_risk_score": float(round(risk_score,2)),
            "distance_from_home": calculated_distance
        }

    except Exception as e:
        return {"error": str(e)}