import os
import sys
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from config import PROCESSED_DATA_PATH
from src.logger import logger


class ModelTrainer:

    def __init__(self):
        self.data_path = PROCESSED_DATA_PATH

    def train(self):

        logger.info("=" * 60)
        logger.info("MODEL TRAINING STARTED")
        logger.info("=" * 60)

        df = pd.read_csv(self.data_path)

        X = df.drop("Churn", axis=1)
        y = df["Churn"]

        numerical_columns = X.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        categorical_columns = X.select_dtypes(
            include=["object"]
        ).columns.tolist()

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    StandardScaler(),
                    numerical_columns
                ),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore"),
                    categorical_columns
                )
            ]
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "XGBoost": XGBClassifier(
                eval_metric="logloss",
                random_state=42
            )
        }

        best_model = None
        best_accuracy = 0

        mlflow.set_experiment("Customer Churn Prediction")

        for name, model in models.items():

            with mlflow.start_run(run_name=name):

                model.fit(X_train, y_train)

                predictions = model.predict(X_test)

                accuracy = accuracy_score(y_test, predictions)
                precision = precision_score(y_test, predictions)
                recall = recall_score(y_test, predictions)
                f1 = f1_score(y_test, predictions)

                print("\n", "=" * 60)
                print(name)
                print("=" * 60)
                print("Accuracy :", accuracy)
                print("Precision:", precision)
                print("Recall   :", recall)
                print("F1 Score :", f1)

                mlflow.log_param("model", name)

                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)

                # Log model
# Log model only for sklearn models
            if name != "XGBoost":
                 mlflow.sklearn.log_model(
                     sk_model=model,
                     name="model"
                )

            if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model



        os.makedirs("models", exist_ok=True)

        joblib.dump(
            best_model,
            "models/best_model.pkl"
        )

        joblib.dump(
            preprocessor,
            "models/preprocessor.pkl"
        )

        print("\nBest Model Saved Successfully")
        print("Preprocessor Saved Successfully")

def main():

    trainer = ModelTrainer()

    trainer.train()


if __name__ == "__main__":
    main()