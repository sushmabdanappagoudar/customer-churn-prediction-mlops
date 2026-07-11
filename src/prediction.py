import os
import sys
import joblib
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.logger import logger


class CustomerChurnPrediction:

    def __init__(self):

        logger.info("Loading Model...")

        self.model = joblib.load("models/best_model.pkl")

        self.preprocessor = joblib.load(
            "artifacts/preprocessor.pkl"
        )

    def predict(self, customer_data):

        logger.info("Prediction Started")

        input_df = pd.DataFrame([customer_data])

        transformed_data = self.preprocessor.transform(input_df)

        prediction = self.model.predict(transformed_data)

        probability = None

        if hasattr(self.model, "predict_proba"):
            probability = self.model.predict_proba(
                transformed_data
            )[0][1]

        if prediction[0] == 1:
            result = "Customer Will Churn"
        else:
            result = "Customer Will Stay"

        return {

            "Prediction": result,

            "Probability": float(probability) if probability is not None else None

        }


def main():

    sample_customer = {

        "gender": "Female",

        "SeniorCitizen": 0,

        "Partner": "Yes",

        "Dependents": "No",

        "tenure": 12,

        "PhoneService": "Yes",

        "MultipleLines": "No",

        "InternetService": "Fiber optic",

        "OnlineSecurity": "No",

        "OnlineBackup": "Yes",

        "DeviceProtection": "No",

        "TechSupport": "No",

        "StreamingTV": "Yes",

        "StreamingMovies": "No",

        "Contract": "Month-to-month",

        "PaperlessBilling": "Yes",

        "PaymentMethod": "Electronic check",

        "MonthlyCharges": 75.50,

        "TotalCharges": 900.00

    }

    predictor = CustomerChurnPrediction()

    result = predictor.predict(sample_customer)

    print("\nPrediction Result\n")

    print(result)


if __name__ == "__main__":

    main()