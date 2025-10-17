import streamlit as st
import requests
import json

# -------------------------------
# Streamlit Web App for Credit Risk Prediction
# -------------------------------

st.set_page_config(page_title="Credit Risk Prediction", page_icon="💳", layout="centered")

st.title("💳 Credit Risk Prediction App")
st.markdown("This app predicts whether a credit card client is **High Risk** or **Low Risk** based on financial data.")

# FastAPI endpoint
API_URL = "http://127.0.0.1:8000/predict"

# -------------------------------
# Input Fields
# -------------------------------
st.subheader("🧾 Client Information")

col1, col2 = st.columns(2)
with col1:
    LIMIT_BAL = st.number_input("Credit Limit (LIMIT_BAL)", min_value=0, value=20000, step=1000)
    SEX = st.selectbox("Sex (1 = Male, 2 = Female)", [1, 2])
    EDUCATION = st.selectbox("Education (1=Graduate, 2=University, 3=High School, 4=Others)", [1, 2, 3, 4])
    MARRIAGE = st.selectbox("Marriage (1=Married, 2=Single, 3=Others)", [1, 2, 3])
    AGE = st.number_input("Age", min_value=18, max_value=100, value=30)

with col2:
    PAY_0 = st.selectbox("Repayment Status (Last Month: PAY_0)", [-2, -1, 0, 1, 2])
    BILL_AMT1 = st.number_input("Bill Amount (BILL_AMT1)", min_value=0, value=5000, step=500)
    PAY_AMT1 = st.number_input("Payment Amount (PAY_AMT1)", min_value=0, value=2000, step=100)
    BILL_AMT2 = st.number_input("Bill Amount (BILL_AMT2)", min_value=0, value=3000, step=500)
    PAY_AMT2 = st.number_input("Payment Amount (PAY_AMT2)", min_value=0, value=1000, step=100)

# Hidden defaults for missing features
extra_features = {
    "PAY_2": PAY_0,
    "PAY_3": 0,
    "PAY_4": 0,
    "PAY_5": 0,
    "PAY_6": 0,
    "BILL_AMT3": 0,
    "BILL_AMT4": 0,
    "BILL_AMT5": 0,
    "BILL_AMT6": 0,
    "PAY_AMT3": 0,
    "PAY_AMT4": 0,
    "PAY_AMT5": 0,
    "PAY_AMT6": 0
}

# Collect all data into one dictionary
data = {
    "LIMIT_BAL": LIMIT_BAL,
    "SEX": SEX,
    "EDUCATION": EDUCATION,
    "MARRIAGE": MARRIAGE,
    "AGE": AGE,
    "PAY_0": PAY_0,
    "BILL_AMT1": BILL_AMT1,
    "BILL_AMT2": BILL_AMT2,
    "PAY_AMT1": PAY_AMT1,
    "PAY_AMT2": PAY_AMT2,
    **extra_features
}

# -------------------------------
# Submit Button
# -------------------------------
if st.button("🔍 Predict Risk"):
    try:
        response = requests.post(API_URL, data=json.dumps(data))
        if response.status_code == 200:
            result = response.json()
            risk = result["risk"]

            if risk == "High Risk":
                st.error(f"🚨 Prediction: {risk}")
            else:
                st.success(f"✅ Prediction: {risk}")

            st.write("Model raw output:", result)
        else:
            st.warning(f"⚠️ Server returned status code {response.status_code}")
    except Exception as e:
        st.error(f"❌ Error connecting to API: {e}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("Made with ❤️ by Jingyi Wang — Powered by FastAPI & Streamlit")
