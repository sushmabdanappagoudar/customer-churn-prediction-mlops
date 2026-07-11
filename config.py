import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


RAW_DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

PROCESSED_DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "customer_churn.csv"
)


MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "xgboost_model.pkl"
)


LOG_DIR = os.path.join(BASE_DIR, "logs")

LOG_FILE = os.path.join(
    LOG_DIR,
    "training.log"
)