import os
import sys
import pandas as pd

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import RAW_DATA_PATH
from src.logger import logger


class DataIngestion:

    def __init__(self):
        self.file_path = RAW_DATA_PATH

    def load_data(self):

        logger.info("=" * 60)
        logger.info("DATA INGESTION STARTED")
        logger.info("=" * 60)

        # Check if dataset exists
        if not os.path.exists(self.file_path):
            logger.error("Dataset not found.")
            raise FileNotFoundError(
                f"Dataset not found:\n{self.file_path}"
            )

        try:
            # Try reading as comma-separated
            df = pd.read_csv(self.file_path)

            # If only one column is found, try tab-separated
            if df.shape[1] == 1:
                logger.warning("Dataset appears to be tab-separated. Reading again...")
                df = pd.read_csv(self.file_path, sep="\t")

            logger.info("Dataset Loaded Successfully")

            return df

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise


def main():

    ingestion = DataIngestion()

    df = ingestion.load_data()

    print("\nFirst 5 Rows\n")
    print(df.head())

    print("\nShape :", df.shape)

    print("\nColumns:\n")
    print(df.columns.tolist())


if __name__ == "__main__":
    main()