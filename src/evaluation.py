import os
import sys
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import PROCESSED_DATA_PATH
from src.logger import logger
from src.feature_engineering import FeatureEngineering


class ModelEvaluation:

    def __init__(self):
        self.data_path = PROCESSED_DATA_PATH

    def evaluate(self):

        logger.info("=" * 60)
        logger.info("MODEL EVALUATION STARTED")
        logger.info("=" * 60)

        # Get transformed data
        feature_engineering = FeatureEngineering()

        X_train, X_test, y_train, y_test = feature_engineering.process()

        # Load best model
        model = joblib.load("models/best_model.pkl")

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        try:
            probabilities = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, probabilities)
        except:
            roc_auc = "Not Available"

        print("\n")
        print("=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)

        print(f"Accuracy  : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1 Score  : {f1:.4f}")

        if roc_auc != "Not Available":
            print(f"ROC AUC   : {roc_auc:.4f}")

        print("\nClassification Report\n")

        print(classification_report(y_test, predictions))

        # Confusion Matrix
        cm = confusion_matrix(y_test, predictions)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm
        )

        disp.plot()

        os.makedirs("artifacts", exist_ok=True)

        plt.savefig("artifacts/confusion_matrix.png")

        plt.close()

        logger.info("Evaluation Completed")

        print("\nConfusion Matrix Saved")

        return accuracy


def main():

    evaluator = ModelEvaluation()

    evaluator.evaluate()


if __name__ == "__main__":
    main()