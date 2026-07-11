import os
import sys
import joblib
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from config import PROCESSED_DATA_PATH
from src.logger import logger


class FeatureEngineering:

    def __init__(self):
        self.data_path = PROCESSED_DATA_PATH

    def process(self):

        logger.info("=" * 60)
        logger.info("FEATURE ENGINEERING STARTED")
        logger.info("=" * 60)

        # Load Processed Dataset
        df = pd.read_csv(self.data_path)

        print("\nProcessed Dataset Shape")
        print(df.shape)

        # ----------------------------
        # Separate Features and Target
        # ----------------------------

        X = df.drop("Churn", axis=1)
        y = df["Churn"]

        print("\nFeature Shape :", X.shape)
        print("Target Shape :", y.shape)

        # ----------------------------
        # Numerical Columns
        # ----------------------------

        numerical_columns = X.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        # ----------------------------
        # Categorical Columns
        # ----------------------------

        categorical_columns = X.select_dtypes(
            include=["object"]
        ).columns.tolist()

        print("\nNumerical Columns")
        print(numerical_columns)

        print("\nCategorical Columns")
        print(categorical_columns)

        # ----------------------------
        # Numerical Pipeline
        # ----------------------------

        numerical_pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler())
            ]
        )

        # ----------------------------
        # Categorical Pipeline
        # ----------------------------

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]
        )

        # ----------------------------
        # Column Transformer
        # ----------------------------

        preprocessor = ColumnTransformer(

            transformers=[

                (
                    "num",
                    numerical_pipeline,
                    numerical_columns
                ),

                (
                    "cat",
                    categorical_pipeline,
                    categorical_columns
                )

            ]

        )

        # ----------------------------
        # Train Test Split
        # ----------------------------

        X_train, X_test, y_train, y_test = train_test_split(

            X,
            y,

            test_size=0.20,

            random_state=42,

            stratify=y

        )

        print("\nTraining Shape")

        print(X_train.shape)

        print(y_train.shape)

        print("\nTesting Shape")

        print(X_test.shape)

        print(y_test.shape)

        # ----------------------------
        # Fit Transformer
        # ----------------------------

        X_train = preprocessor.fit_transform(X_train)

        X_test = preprocessor.transform(X_test)

        # ----------------------------
        # Save Transformer
        # ----------------------------

        os.makedirs("artifacts", exist_ok=True)

        joblib.dump(
            preprocessor,
            "artifacts/preprocessor.pkl"
        )

        logger.info("Preprocessor Saved")

        print("\nFeature Engineering Completed")

        return (

            X_train,

            X_test,

            y_train,

            y_test

        )


def main():

    engineering = FeatureEngineering()

    engineering.process()


if __name__ == "__main__":
    main()