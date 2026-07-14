# Customer Churn Prediction using MLOps

An end-to-end **Machine Learning Operations (MLOps)** project that predicts customer churn for a telecom company. The project demonstrates the complete ML lifecycle, including data preprocessing, model training, experiment tracking with MLflow, API deployment using FastAPI, and Docker containerization.

## 🚀 Live Demo

**Try the application here:**

**🔗 https://sushmabdanappagoudar-customer-churn-predic-streamlit-app-my4fsi.streamlit.app**

---

## Features

- Customer churn prediction using Machine Learning
- Data preprocessing and feature engineering
- Model training and evaluation
- Experiment tracking with MLflow
- REST API using FastAPI
- Docker containerization
- Modular and scalable project structure

---

## Tech Stack

| Category | Technologies |
|----------|--------------|
| Language | Python 3.13 |
| ML | Scikit-learn, XGBoost |
| Data | Pandas, NumPy |
| API | FastAPI, Uvicorn |
| MLOps | MLflow |
| Deployment | Docker, Docker Compose |

---

## Project Structure

```text
Customer-Churn-Prediction-MLOps/
│
├── app/
├── data/
├── src/
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── prediction.py
│
├── application.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Workflow

```
Dataset
   ↓
Data Ingestion
   ↓
Data Preprocessing
   ↓
Feature Engineering
   ↓
Model Training
   ↓
Model Evaluation
   ↓
MLflow Tracking
   ↓
FastAPI Deployment
   ↓
Docker
```

---

## Dataset

The model is trained on the **Telecom Customer Churn Dataset**, which includes customer demographics, subscription details, service usage, and churn status.

---

## Model Performance

| Metric | Score |
|--------|------:|
| Accuracy | **80.5%** |
| Precision | **65.7%** |
| Recall | **55.9%** |
| F1-Score | **60.4%** |

---

## Installation

Clone the repository

```bash
git clone https://github.com/sushmabdanappagoudar/customer-churn-prediction-mlops.git
cd customer-churn-prediction-mlops
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment

**Windows**

```bash
.venv\Scripts\activate
```

**Linux/macOS**

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Application

Start the FastAPI server

```bash
uvicorn application:app --reload
```

API Documentation

```
http://localhost:8000/docs
```

Start MLflow

```bash
mlflow ui
```

MLflow UI

```
http://127.0.0.1:5000
```

---

## Docker

Build the Docker image

```bash
docker build -t customer-churn-api .
```

Run the container

```bash
docker run -p 8000:8000 customer-churn-api
```

---

## Sample Prediction

### Request

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 10,
  "PhoneService": "Yes",
  "InternetService": "Fiber optic",
  "MonthlyCharges": 75.5,
  "TotalCharges": 755.0
}
```

### Response

```json
{
  "prediction": "No Churn"
}
```

---

## Future Improvements

- Hyperparameter tuning
- CI/CD with GitHub Actions
- Kubernetes deployment
- Automated model retraining
- Model monitoring

---

## Author

**Sushma B Danappagoudar**

Computer Science Engineering (Data Science)

- GitHub: https://github.com/sushmabdanappagoudar

---

## License

This project is developed for educational purposes.