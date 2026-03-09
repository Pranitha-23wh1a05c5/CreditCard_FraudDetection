import streamlit as st
import requests

st.title("Credit Card Fraud Detection AI Agent")

st.write("Enter transaction details")

# ---------------- AMOUNT ---------------- #

amount = st.number_input("Transaction Amount")

# ---------------- PRODUCT ---------------- #

product_map = {
    "Electronics": 0,
    "Retail": 1,
    "Groceries": 2,
    "Travel": 3,
    "Subscription": 4
}

product_choice = st.selectbox("Product Type", list(product_map.keys()))
product_code = product_map[product_choice]


# ---------------- CARD TYPE ---------------- #

card_map = {
    "Visa": 15000,
    "Mastercard": 13000,
    "American Express": 17000,
    "Discover": 12000
}

card_choice = st.selectbox("Card Type", list(card_map.keys()))
card1 = card_map[card_choice]

card2 = st.number_input("Card Bank")


# ---------------- COUNTRY DATA ---------------- #

countries = {
    356: "India",
    840: "USA",
    826: "UK",
    124: "Canada",
    36: "Australia",
    566: "Nigeria",
    156: "China",
    643: "Russia"
}

regions = {
    356: {"Andhra Pradesh":101,"Telangana":102,"Karnataka":103},
    840: {"California":201,"Texas":202,"New York":203},
    826: {"England":301,"Scotland":302,"Wales":303},
    124: {"Ontario":401,"Quebec":402,"British Columbia":403},
    36: {"New South Wales":501,"Victoria":502},
    566: {"Lagos":601,"Abuja":602}
}


# ---------------- COUNTRY INPUT ---------------- #

addr2 = st.number_input("Country Code")

country_name = countries.get(int(addr2), None)

if country_name:
    st.success(f"{addr2} - {country_name}")
else:
    if addr2 != 0:
        st.warning("Country code not recognized")


# ---------------- REGION DROPDOWN ---------------- #

addr1 = None

if country_name:

    region_dict = regions.get(int(addr2), {})

    if region_dict:

        region_choice = st.selectbox(
            "Billing Region",
            list(region_dict.keys())
        )

        addr1 = region_dict[region_choice]

    else:
        st.info("No regions available for this country")


st.info("Distance from home will be calculated automatically")


# ---------------- FRAUD CHECK ---------------- #

if st.button("Check Fraud"):

    if not country_name or addr1 is None:
        st.error("Please enter valid country code and region")
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
            status = result["prediction"]
            distance = result["distance_from_home"]

            st.subheader("Prediction Result")

            if status == "Fraud":
                st.error(f"🚨 Fraud Transaction! Risk Score: {risk}%")

            elif status == "Suspicious":
                st.warning(f"⚠ Suspicious Transaction! Risk Score: {risk}%")

            else:
                st.success(f"✅ Safe Transaction. Risk Score: {risk}%")

            st.progress(risk / 100)

            st.subheader("Transaction Details")

            st.write("Country:", f"{addr2} - {country_name}")
            st.write("Region:", region_choice)
            st.write("Calculated Distance:", distance, "km")

             # -------- FRAUD EXPLANATION -------- #

            st.subheader("Fraud Detection Explanation")

            reasons = []

            if amount > 2000:
                reasons.append("⚠ High transaction amount")

            if distance > 5000:
                reasons.append("⚠ Large distance from home")

            if addr2 != 356:
                reasons.append("⚠ Foreign country transaction")

            very_high_risk_countries = {566,231,694,706,332,140,180}
            high_risk_countries = {643,156,586,50,360,608,704,710,404,818}

            if addr2 in very_high_risk_countries:
                reasons.append("🚨 Very high fraud risk country")

            elif addr2 in high_risk_countries:
                reasons.append("⚠ High fraud risk country")

            if reasons:
                for r in reasons:
                    st.write(r)
            else:
                st.write("No major fraud indicators detected.")

        else:
            st.error("API returned an error")
            st.write(response.text)