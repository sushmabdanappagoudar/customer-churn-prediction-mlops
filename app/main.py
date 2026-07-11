from fastapi import FastAPI
from app.schema import CustomerData
from src.prediction import CustomerChurnPrediction

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn using a trained ML model.",
    version="1.0.0"
)

predictor = CustomerChurnPrediction()


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is Running!"
    }


@app.post("/predict")
def predict(customer: CustomerData):

    result = predictor.predict(customer.dict())

    return result