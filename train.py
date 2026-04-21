"""
train.py - Train a RandomForest model on the Titanic dataset with MLflow tracking.
Logs parameters, metrics, and artifacts, then registers the model in MLflow Model Registry.
"""

import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

# ──────────────────────────────────────────────
# 1. LOAD AND PREPROCESS DATA
# ──────────────────────────────────────────────

def load_titanic_data() -> pd.DataFrame:
    """Load the Titanic dataset from a local CSV file."""
    df = pd.read_csv("/Users/aruzhantleukul/ml-fastapi-docker/Titanic-Dataset.csv")
    return df


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Select relevant features, fill missing values, encode categoricals.
    Returns feature matrix X and target vector y.
    """
    df = df.copy()

    # Select features used for prediction
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    target = "Survived"

    df = df[features + [target]].copy()

    # Fill missing values
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Fare"].fillna(df["Fare"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

    # Encode categorical columns
    le = LabelEncoder()
    df["Sex"] = le.fit_transform(df["Sex"])          # male=1, female=0
    df["Embarked"] = le.fit_transform(df["Embarked"]) # C=0, Q=1, S=2

    X = df[features]
    y = df[target]
    return X, y


# ──────────────────────────────────────────────
# 2. TRAINING WITH MLFLOW
# ──────────────────────────────────────────────

def train():
    # Model hyperparameters
    PARAMS = {
        "n_estimators": 100,
        "max_depth": 6,
        "min_samples_split": 4,
        "random_state": 42,
    }

    # Load and split data
    df = load_titanic_data()
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Point MLflow to the local tracking server (or use default ./mlruns)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # Create / select experiment
    mlflow.set_experiment("Titanic_Survival_Prediction")

    with mlflow.start_run(run_name="RandomForest_v1") as run:
        # ── Train ──────────────────────────────
        model = RandomForestClassifier(**PARAMS)
        model.fit(X_train, y_train)

        # ── Evaluate ───────────────────────────
        y_pred = model.predict(X_test)
        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred)

        print(f"[INFO] Accuracy : {acc:.4f}")
        print(f"[INFO] F1-score : {f1:.4f}")

        # ── Log parameters ─────────────────────
        mlflow.log_params(PARAMS)

        # ── Log metrics ────────────────────────
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # ── Save model artifact (joblib) ────────
        joblib.dump(model, "model.joblib")
        mlflow.log_artifact("model.joblib")

        # ── Log model + register in Model Registry
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="sklearn-model",
            registered_model_name="TitanicModel",
        )

        print(f"[INFO] Run ID   : {run.info.run_id}")
        print("[INFO] Model registered as 'TitanicModel' in MLflow Model Registry.")


if __name__ == "__main__":
    train()
