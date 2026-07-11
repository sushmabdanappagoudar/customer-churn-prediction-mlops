import os
import sys
import pandas as pd

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import PROCESSED_DATA_PATH
from src.data_ingestion import DataIngestion
from src.logger import logger


class DataPreprocessing:

    def __init__(self):
        self.ingestion = DataIngestion()

    def preprocess(self):

        logger.info("=" * 60)
        logger.info("DATA PREPROCESSING STARTED")
        logger.info("=" * 60)

        # ----------------------------
        # Load Dataset
        # ----------------------------

        df = self.ingestion.load_data()

        print("\nOriginal Shape :", df.shape)

        print("\nColumns Before Preprocessing:")
        print(df.columns.tolist())

        # ----------------------------
        # Remove customerID
        # ----------------------------

        if "customerID" in df.columns:
            logger.info("Removing customerID column")
            df.drop(columns=["customerID"], inplace=True)
        else:
            logger.warning("customerID column already removed.")

        # ----------------------------
        # Convert TotalCharges
        # ----------------------------

        if "TotalCharges" in df.columns:

            logger.info("Converting TotalCharges to numeric")

            df["TotalCharges"] = pd.to_numeric(
                df["TotalCharges"],
                errors="coerce"
            )

            logger.info("Filling Missing Values")

            df["TotalCharges"] = df["TotalCharges"].fillna(
                df["TotalCharges"].median()
            )

        # ----------------------------
        # Encode Target
        # ----------------------------

        if "Churn" in df.columns:

            logger.info("Encoding Churn Column")

            if df["Churn"].dtype == "object":

                df["Churn"] = df["Churn"].map(
                    {
                        "Yes": 1,
                        "No": 0
                    }
                )

        print("\nProcessed Shape :", df.shape)

        print("\nMissing Values")
        print(df.isnull().sum())

        print("\nData Types")
        print(df.dtypes)

        # ----------------------------
        # Save Processed Dataset
        # ----------------------------

        os.makedirs(
            os.path.dirname(PROCESSED_DATA_PATH),
            exist_ok=True
        )

        df.to_csv(
            PROCESSED_DATA_PATH,
            index=False
        )

        logger.info("Processed Dataset Saved Successfully")

        print("\nProcessed dataset saved at:")
        print(PROCESSED_DATA_PATH)

        return df


def main():

    preprocessing = DataPreprocessing()

    preprocessing.preprocess()


if __name__ == "__main__":
    main()