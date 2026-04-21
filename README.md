# Titanic ML System - FastAPI + MLflow + Streamlit + Docker

This project demonstrates an end-to-end machine learning workflow for predicting passenger survival on the Titanic dataset. 

The system includes model training, experiment tracking, API-based model serving, and a simple web interface for interaction. The project also demonstrates basic MLOps practices such as model version tracking and containerized deployment.

---
## Project Overview

The goal of this project is to develop a machine learning system capable of predicting whether a passenger survived the Titanic disaster based on demographic and travel-related features.

The system demonstrates the following components of a typical ML pipeline:

- data acquisition and preprocessing
- model training and evaluation
- experiment tracking with MLflow
- deployment of the trained model using FastAPI
- user interaction through a Streamlit interface
- containerized execution using Docker
---

##  Project Structure

```
ml-fastapi-docker/
├── train.py          # Train model, log to MLflow, register in Model Registry
├── main.py           # FastAPI application (port 1912)
├── app.py            # Streamlit frontend
├── model.joblib      # Saved model artifact (generated after training)
├── requirements.txt  # Python dependencies
├── Dockerfile        # Container instructions
└── README.md         # This file
```

---
## Technologies Used

The following technologies were used in the development of this project:

- Python 3.11
- scikit-learn - machine learning algorithms
- MLflow - experiment tracking and model registry
- FastAPI - REST API for model inference
- Streamlit - interactive user interface
- Docker - containerization and reproducible deployment

---

## Step-by-Step Setup

### Step 1 - Install dependencies

```bash
pip install -r requirements.txt
```

---

### Step 2 - Start the MLflow Tracking Server

Open a **dedicated terminal** and run:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Keep this terminal open. The MLflow UI will be available at:
 http://127.0.0.1:5000

---

### Step 3 - Train the model

In a **new terminal**, run:

```bash
python train.py
```

This will:
- Download the Titanic dataset automatically
- Train a RandomForest classifier
- Log parameters, accuracy, and F1-score to MLflow
- Save `model.joblib` locally
- Register the model as **TitanicModel** in MLflow Model Registry

---

### Step 4 - Run the FastAPI server locally

```bash
uvicorn main:app --host 0.0.0.0 --port 1912 --reload
```

Test endpoints:
| Endpoint | Description |
|---|---|
| http://localhost:1912/ | Health check |
| http://localhost:1912/predict | POST prediction |
| http://localhost:1912/docs | Swagger UI |

---

### Step 5 - Run the Streamlit frontend

```bash
streamlit run app.py
```

👉 Opens at http://localhost:8501

---

### Step 6 - Build the Docker image

```bash
docker build -t titanic-ml-api .
```

---

### Step 7 - Run the Docker container

```bash
docker run -d -p 1912:1912 --name titanic-api titanic-ml-api
```

Test from inside Docker:
```bash
# Health check
curl http://localhost:1912/

# Prediction request
curl -X POST http://localhost:1912/predict \
  -H "Content-Type: application/json" \
  -d '{"Pclass":3,"Sex":1,"Age":22,"SibSp":1,"Parch":0,"Fare":7.25,"Embarked":2}'
```

---

##  Example /predict Payload

```json
{
  "Pclass": 1,
  "Sex": 0,
  "Age": 29.0,
  "SibSp": 0,
  "Parch": 0,
  "Fare": 211.34,
  "Embarked": 0
}
```

**Field encoding:**
| Field | Values |
|---|---|
| Pclass | 1, 2, 3 |
| Sex | 0 = female, 1 = male |
| Embarked | 0 = C, 1 = Q, 2 = S |

---

##  MLflow Tracking

- Experiment: `Titanic_Survival_Prediction`
- Logged: `n_estimators`, `max_depth`, `min_samples_split`, `accuracy`, `f1_score`
- Artifact: `model.joblib`
- Registry name: `TitanicModel`

---

##  Stop & Clean Up

```bash
# Stop the container
docker stop titanic-api

# Remove the container
docker rm titanic-api

# Remove the image (optional)
docker rmi titanic-ml-api
```
