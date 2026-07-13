import streamlit as st
import requests

API_URL = "https://customer-churn-prediction-mlops.onrender.com/predict"

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Prediction")
st.write("Fill in the customer details and click Predict.")

gender = st.selectbox("Gender", ["Female", "Male"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])

tenure = st.number_input(
    "Tenure (Months)",
    min_value=0,
    max_value=100,
    value=12
)

phone = st.selectbox("Phone Service", ["Yes", "No"])
multiple = st.selectbox("Multiple Lines", ["Yes", "No"])
internet = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

security = st.selectbox("Online Security", ["Yes", "No"])
backup = st.selectbox("Online Backup", ["Yes", "No"])
device = st.selectbox("Device Protection", ["Yes", "No"])
support = st.selectbox("Tech Support", ["Yes", "No"])
tv = st.selectbox("Streaming TV", ["Yes", "No"])
movies = st.selectbox("Streaming Movies", ["Yes", "No"])

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

payment = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly = st.number_input("Monthly Charges", value=70.0)
total = st.number_input("Total Charges", value=900.0)

if st.button("Predict"):

    payload = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": device,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()

            st.success(result["Prediction"])
            st.metric(
                "Probability",
                f"{result['Probability']:.2%}"
            )

        else:
            st.error(f"API Error: {response.status_code}")
            st.write(response.text)

    except Exception as e:
        st.error(str(e))