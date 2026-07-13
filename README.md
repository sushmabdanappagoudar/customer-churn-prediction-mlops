# Customer Churn Prediction using MLOps

## Overview

Customer Churn Prediction is an end-to-end Machine Learning project that predicts whether a customer is likely to leave a telecom company based on customer demographics, subscription details, and service usage.

This project follows MLOps best practices by implementing data preprocessing, model training, experiment tracking with MLflow, API deployment using FastAPI, Docker containerization, and cloud deployment readiness.

---

## Features

- Customer churn prediction using Machine Learning
- Data preprocessing and feature engineering
- Model training and evaluation
- MLflow experiment tracking
- Model version management
- FastAPI REST API
- Docker containerization
- Cloud deployment ready
- Modular project structure

---

## Tech Stack

### Programming Language
- Python 3.13

### Machine Learning
- Scikit-learn
- XGBoost

### Data Processing
- Pandas
- NumPy

### API
- FastAPI
- Uvicorn

### MLOps
- MLflow

### Deployment
- Docker
- Docker Compose
- AWS (Deployment Ready)

---

## Project Structure

```
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
├── config.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── README.md
└── .gitignore
```

---

## Machine Learning Workflow

```
Dataset
   │
   ▼
Data Ingestion
   │
   ▼
Data Preprocessing
   │
   ▼
Feature Engineering
   │
   ▼
Model Training
   │
   ▼
Model Evaluation
   │
   ▼
MLflow Tracking
   │
   ▼
Model Registry
   │
   ▼
FastAPI Deployment
   │
   ▼
Docker Container
```

---

## Dataset

The project uses the Telecom Customer Churn dataset containing customer information such as:

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Internet Service
- Contract Type
- Monthly Charges
- Total Charges
- Churn Status

---

## Model Performance

| Metric | Score |
|---------|-------|
| Accuracy | 80.5% |
| Precision | 65.7% |
| Recall | 55.9% |
| F1 Score | 60.4% |

---

## Installation

Clone the repository

```bash
git clone https://github.com/sushmabdanappagoudar/customer-churn-prediction-mlops.git
```

Move into the project

```bash
cd customer-churn-prediction-mlops
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate virtual environment

Windows

```bash
.venv\Scripts\activate
```

Linux / Mac

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

Run FastAPI

```bash
python application.py
```

or

```bash
uvicorn application:app --reload
```

API Documentation

```
http://localhost:8000/docs
```

---

## MLflow

Start MLflow

```bash
mlflow ui
```

Open

```
http://127.0.0.1:5000
```

MLflow provides:

- Experiment Tracking
- Parameter Logging
- Metrics Logging
- Model Versioning
- Model Comparison

---

## Docker

Build Docker Image

```bash
docker build -t customer-churn-api .
```

Run Docker Container

```bash
docker run -p 8000:8000 customer-churn-api
```

---

## API Example

### POST

```
/predict
```

Sample Input

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

Sample Output

```json
{
    "prediction": "No Churn"
}
```

---

## Future Improvements

- Hyperparameter tuning
- CI/CD pipeline using GitHub Actions
- Kubernetes deployment
- Automated model retraining
- Monitoring using Prometheus & Grafana
- Cloud deployment on AWS

---

## Learning Outcomes

This project demonstrates practical knowledge of:

- Machine Learning
- Data Preprocessing
- Model Evaluation
- FastAPI
- Docker
- MLflow
- MLOps Workflow
- API Development
- Model Deployment

---

## Author

**Sushma B Danappagoudar**

Computer Science Engineering (Data Science)

GitHub: https://github.com/sushmabdanappagoudar

LinkedIn: *(Add your LinkedIn profile here)*

---

## License

This project is developed for educational and learning purposes.