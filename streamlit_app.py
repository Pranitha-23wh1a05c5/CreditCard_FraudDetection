import streamlit as st
import requests
import pycountry
import json

st.title("💳 Credit Card Fraud Detection AI Agent")
st.write("Enter transaction details")


# 💰 Amount
amount = st.number_input("Transaction Amount", min_value=0.0)


# 🛒 Product Mapping
product_map = {
    "Electronics": 0,
    "Retail": 1,
    "Groceries": 2,
    "Travel": 3,
    "Subscription": 4
}

product_choice = st.selectbox("Product Type", list(product_map.keys()))
product_code = product_map[product_choice]


# 💳 Card Mapping
card_map = {
    "Visa": 15000,
    "Mastercard": 13000,
    "American Express": 17000,
    "Discover": 12000
}

card_choice = st.selectbox("Card Type", list(card_map.keys()))
card1 = card_map[card_choice]

card2 = st.number_input("Card Bank")


# 🌍 LOAD COUNTRY → STATES DATASET (FIXED VERSION)
@st.cache_data
def load_country_states():
    with open("countries_states.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    country_state_map = {}

    # ✅ Case 1: list format (most common)
    if isinstance(data, list):
        for country in data:
            name = country.get("name")

            raw_states = country.get("states", [])

            states = []
            for state in raw_states:
                if isinstance(state, dict):
                    states.append(state.get("name"))
                elif isinstance(state, str):
                    states.append(state)

            if states:
                country_state_map[name] = states

    # ✅ Case 2: dict format
    elif isinstance(data, dict):
        for country, states in data.items():
            country_state_map[country] = states

    return country_state_map


country_regions = load_country_states()


# 🌍 Country Selection
countries = {}

for country in pycountry.countries:
    if hasattr(country, "numeric"):
        countries[int(country.numeric)] = country.name

country_names = list(countries.values())
country_codes = list(countries.keys())

country_choice = st.selectbox("Select Country", country_names)

addr2 = country_codes[country_names.index(country_choice)]

st.success(f"{addr2} - {country_choice}")


# 📍 REAL REGION DROPDOWN (SAFE + SEARCHABLE)
regions_list = country_regions.get(country_choice)

if regions_list:
    region_choice = st.selectbox(
        "Billing Region",
        regions_list,
        index=None,
        placeholder="Search or select a state..."
    )
else:
    st.warning("No state data available for this country")
    region_choice = None


# 🔢 SAFE ENCODING (prevents crash)
if region_choice:
    addr1 = abs(hash(region_choice + country_choice)) % 10000
else:
    addr1 = 0


st.info("Distance from home will be calculated automatically")


# 🚀 Predict Button
if st.button("Check Fraud"):

    # ❗ Optional validation (nice UX)
    if not region_choice:
        st.error("Please select a billing region before proceeding")
    else:

        data = {
            "TransactionAmt": amount,
            "ProductCD": product_code,
            "card1": card1,
            "card2": card2,
            "addr1": addr1,
            "addr2": addr2
        }

        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=data
        )

        if response.status_code == 200:

            result = response.json()

            risk = result["fraud_risk_score"]
            prob = result["fraud_probability"]
            status = result["prediction"]
            distance = result["distance_from_home"]
            agent_analysis = result["agent_analysis"]

            st.subheader("Prediction Result")

            if status == "Fraud":
                st.error(f"🚨 Fraud Transaction! Risk Score: {risk}%")
            elif status == "Suspicious":
                st.warning(f"⚠ Suspicious Transaction! Risk Score: {risk}%")
            else:
                st.success(f"✅ Safe Transaction. Risk Score: {risk}%")

            # 🧠 Model Probability
            st.write(f"🧠 Model Fraud Probability: {prob}%")

            st.progress(risk / 100)


            st.subheader("Transaction Details")

            st.write("Country:", f"{addr2} - {country_choice}")
            st.write("Region:", region_choice)
            st.write("Calculated Distance:", distance, "km")


            st.subheader("🤖 AI Fraud Analyst")
            st.write(agent_analysis)

        else:
            st.error("API returned an error")
            st.write(response.text)