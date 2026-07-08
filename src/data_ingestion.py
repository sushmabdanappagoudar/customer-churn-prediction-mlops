import os
import sys

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd

from config import RAW_DATA_PATH
from src.logger import logger


class DataIngestion:
    """
    Responsible for loading the raw dataset.
    """

    def __init__(self):
        self.file_path = RAW_DATA_PATH

    def load_data(self):

        logger.info("=" * 50)
        logger.info("Data Ingestion Started")
        logger.info("=" * 50)

        # Check whether dataset exists
        if not os.path.exists(self.file_path):

            logger.error("Dataset Not Found")

            raise FileNotFoundError(
                f"Dataset not found:\n{self.file_path}"
            )

        logger.info("Dataset Found")

        # Read CSV
        df = pd.read_csv(self.file_path)

        logger.info("Dataset Loaded Successfully")

        return df

    def dataset_summary(self, df):

        logger.info("Generating Dataset Summary")

        print("\n")
        print("=" * 60)
        print("DATASET SHAPE")
        print("=" * 60)
        print(df.shape)

        print("\n")
        print("=" * 60)
        print("COLUMN NAMES")
        print("=" * 60)
        print(df.columns.tolist())

        print("\n")
        print("=" * 60)
        print("DATA TYPES")
        print("=" * 60)
        print(df.dtypes)

        print("\n")
        print("=" * 60)
        print("MISSING VALUES")
        print("=" * 60)
        print(df.isnull().sum())

        print("\n")
        print("=" * 60)
        print("FIRST FIVE ROWS")
        print("=" * 60)
        print(df.head())

        logger.info("Summary Generated")


def main():

    ingestion = DataIngestion()

    df = ingestion.load_data()

    ingestion.dataset_summary(df)


if __name__ == "__main__":
    main()