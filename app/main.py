import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from agent import fraud_agent, parse_agent_output
from geopy.distance import geodesic
from geopy.geocoders import Nominatim

app = FastAPI(
    title="Credit Card Fraud Detection API",
    version="5.0"
)

model = joblib.load("app/ieee_fraud_model.pkl")

geolocator = Nominatim(user_agent="fraud_app")


class Transaction(BaseModel):
    TransactionAmt: float
    ProductCD: int
    card1: int
    card2: float
    addr1: float
    addr2: float
    user_id: str


@app.get("/")
def home():
    return {"message": "Agentic Fraud Detection API 🚀"}


# ✅ SET YOUR HOME LOCATION (Hyderabad)
HOME_COORDS = (17.3850, 78.4867)


@app.post("/predict")
def predict(data: Transaction):

    try:
        # 🌍 Convert country code → name (approx mapping)
        country_code_map = {
            356: "India",
            840: "USA",
            826: "UK",
            124: "Canada",
            36: "Australia"
        }

        country_name = country_code_map.get(int(data.addr2), "Unknown")

        # 🌍 Get coordinates dynamically
        location = geolocator.geocode(country_name)

        if location:
            coords = (location.latitude, location.longitude)
            calculated_distance = int(geodesic(HOME_COORDS, coords).km)
        else:
            calculated_distance = int(geodesic(HOME_COORDS, (0, 0)).km)

        # 📊 Features
        features = pd.DataFrame([{
            "TransactionAmt": data.TransactionAmt,
            "ProductCD": data.ProductCD,
            "card1": data.card1,
            "card2": data.card2,
            "addr1": data.addr1,
            "addr2": data.addr2,
            "dist1": calculated_distance
        }])

        # 🤖 Model
        prob = model.predict_proba(features)[0][1]
        risk_score = prob * 100

        # ⚙ Rules
        if data.TransactionAmt > 2000:
            risk_score += 20

        if calculated_distance > 5000:
            risk_score += 15

        if data.addr2 != 356:
            risk_score += 10

        risk_score = min(risk_score, 100)

        # 🤖 Agent
        model_result = {
            "prediction": "Unknown",
            "risk_score": risk_score,
            "distance": calculated_distance
        }

        agent_output = fraud_agent(data.dict(), model_result)

        decision, action = parse_agent_output(agent_output)

        return {
            "prediction": decision,
            "action": action,
            "fraud_probability": float(round(prob * 100, 2)),
            "fraud_risk_score": float(round(risk_score, 2)),
            "distance_from_home": calculated_distance,
            "agent_analysis": agent_output
        }

    except Exception as e:
        return {"error": str(e)}