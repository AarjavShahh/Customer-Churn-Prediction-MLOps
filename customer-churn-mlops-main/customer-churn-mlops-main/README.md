# Customer Churn Prediction | End-to-End MLOps Project

## 📌 Project Overview

Customer Churn Prediction is an end-to-end Machine Learning project designed to predict whether a customer is likely to leave a telecommunications service provider.

The project uses a **Gradient Boosting Classifier** to analyze customer demographics, account information, subscription details, and service usage patterns to predict customer churn.

The trained Machine Learning model is integrated with **MLflow and FastAPI** to serve real-time predictions through REST APIs. The application is containerized using Docker and monitored using Prometheus to track API performance, prediction requests, errors, and response latency.

The primary objective is to build a scalable and production-oriented Machine Learning system that covers the complete ML lifecycle, from data preprocessing and model development to deployment and monitoring.

## 🎯 Project Objectives

- Analyze customer data to identify churn patterns.
- Perform data inspection and preprocessing.
- Develop a Gradient Boosting classification model.
- Track and manage the production model using MLflow.
- Build REST APIs using FastAPI for real-time predictions.
- Containerize the application using Docker.
- Monitor API performance using Prometheus.
- Track prediction requests, errors, and response latency.

## 🛠️ Technologies & Tools

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| ML Model | Gradient Boosting Classifier |
| Experiment & Model Management | MLflow |
| API Development | FastAPI |
| Data Validation | Pydantic |
| Data Preprocessing | OneHotEncoder, StandardScaler |
| Containerization | Docker |
| Monitoring | Prometheus |
| API Server | Uvicorn |

## 📂 Project Structure

```text
Customer-Churn-Prediction/
│
├── data/
│   └── ML DATASET_Telco-Customer-Churn.csv
│
├── model/
│   └── MLflow production model
│
├── api.py
├── inspect_data.py
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

## 📊 Dataset Description

The project uses the Telco Customer Churn dataset.

The dataset contains customer information related to demographics, subscribed services, account details, and billing information.

### Important Features

| Feature | Description |
|---|---|
| gender | Customer gender |
| SeniorCitizen | Whether the customer is a senior citizen |
| Partner | Whether the customer has a partner |
| Dependents | Whether the customer has dependents |
| tenure | Number of months the customer has stayed |
| PhoneService | Phone service subscription |
| MultipleLines | Multiple phone line subscription |
| InternetService | Internet service type |
| OnlineSecurity | Online security subscription |
| OnlineBackup | Online backup subscription |
| DeviceProtection | Device protection subscription |
| TechSupport | Technical support subscription |
| StreamingTV | Streaming TV subscription |
| StreamingMovies | Streaming movies subscription |
| Contract | Customer contract type |
| PaperlessBilling | Paperless billing preference |
| PaymentMethod | Customer payment method |
| MonthlyCharges | Monthly service charges |
| TotalCharges | Total customer charges |
| Churn | Target variable indicating customer churn |

### Data Inspection

The `inspect_data.py` script performs initial dataset analysis, including:

- Dataset shape and column information.
- Data types.
- Missing value identification.
- Duplicate record identification.
- Churn distribution.
- Churn percentage.
- Unique values in categorical columns.
- Display of the first five records.

## ⚙️ Data Preprocessing

The project uses Scikit-learn's `ColumnTransformer` to process numerical and categorical features.

**Categorical Features:**
- One-Hot Encoding using `OneHotEncoder`.
- `handle_unknown="ignore"` to handle previously unseen categories.

**Numerical Features:**
- Standardization using `StandardScaler`.

Missing values in `TotalCharges` are converted to numeric values and filled with zero.

The preprocessing pipeline is fitted using the historical customer dataset and applied to incoming API requests.

## 🤖 Machine Learning Model

**Selected Algorithm: Gradient Boosting Classifier**

Gradient Boosting is an ensemble learning algorithm that combines multiple weak decision-tree learners to create a stronger predictive model.

The model predicts whether a customer is likely to churn based on the processed customer information.

### Model Management with MLflow

MLflow is used to load and manage the trained production Machine Learning model.

The production model is loaded using:

```python
model = mlflow.sklearn.load_model(MODEL_PATH)
```

The model is loaded from the Docker model directory (`/app/model`) or the local `model` directory.

## 🚀 API Development with FastAPI

FastAPI provides REST API endpoints for customer churn predictions.

### Available API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API information and available endpoints |
| GET | `/health` | Checks API and model health |
| POST | `/predict` | Predicts customer churn |
| GET | `/metrics` | Exposes Prometheus monitoring metrics |
| GET | `/docs` | Interactive API documentation |

### Prediction Output

The prediction API returns:

- **Prediction:** Binary classification result (0 or 1).
- **Churn:** Yes or No.
- **Churn Probability:** Probability of customer churn.

Example response:

```json
{
    "prediction": 1,
    "churn": "Yes",
    "churn_probability": 0.8521
}
```

*The values above are illustrative.*

## 📈 Monitoring with Prometheus

Prometheus monitoring is integrated into the FastAPI application to track application performance.

### Custom Metrics

| Metric | Description |
|---|---|
| prediction_requests_total | Total number of prediction requests |
| prediction_errors_total | Total number of prediction errors |
| prediction_request_latency_seconds | Time taken to process prediction requests |

FastAPI instrumentation is also enabled through `prometheus-fastapi-instrumentator`.

Monitoring metrics are exposed through the `/metrics` endpoint.

## 🐳 Docker Deployment

The application is containerized using Docker to provide a consistent deployment environment.

The Dockerfile:

- Uses Python 3.12 Slim as the base image.
- Installs dependencies from `requirements.txt`.
- Copies the FastAPI application.
- Copies the customer churn dataset.
- Copies the trained MLflow model.
- Exposes port 8000.
- Starts the application using Uvicorn.

### Build Docker Image

```bash
docker build -t customer-churn-api .
```

### Run Docker Container

```bash
docker run -p 8000:8000 customer-churn-api
```

### Access the Application

```text
http://localhost:8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

Health endpoint:

```text
http://localhost:8000/health
```

Prometheus metrics:

```text
http://localhost:8000/metrics
```

## 🔄 End-to-End Project Workflow

1. Load the Telco Customer Churn dataset.
2. Perform data inspection and analysis.
3. Preprocess categorical and numerical features.
4. Train the Gradient Boosting classification model.
5. Manage the trained model using MLflow.
6. Load the production model into FastAPI.
7. Accept customer information through REST API requests.
8. Apply preprocessing to incoming customer data.
9. Generate churn predictions and probabilities.
10. Monitor API requests, errors, and latency using Prometheus.
11. Package the application and model using Docker.

## 🔮 Future Improvements

- Integrate Grafana dashboards for advanced monitoring and visualization.
- Implement automated model retraining.
- Introduce data drift and model drift detection.
- Build a Streamlit frontend for customer churn predictions.
- Implement CI/CD pipelines using GitHub Actions.
- Deploy the application to a cloud platform.
- Improve model performance through hyperparameter optimization.
