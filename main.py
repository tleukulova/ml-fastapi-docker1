"""
main.py - FastAPI application that serves Titanic survival predictions.
Exposes:
  GET  /         → health check
  POST /predict  → survival prediction from passenger features
Runs on port 1912 via Uvicorn.
"""

import os
import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ──────────────────────────────────────────────
# APP INITIALIZATION
# ──────────────────────────────────────────────

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="Predict survival of a Titanic passenger using a trained RandomForest model.",
    version="1.0.0",
)

# Load model once at startup (not on every request)
MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")

try:
    model = joblib.load(MODEL_PATH)
    print(f"[INFO] Model loaded successfully from '{MODEL_PATH}'")
except FileNotFoundError:
    model = None
    print(f"[WARNING] Model file '{MODEL_PATH}' not found. Run train.py first.")


# ──────────────────────────────────────────────
# SCHEMAS
# ──────────────────────────────────────────────

class PassengerFeatures(BaseModel):
    """Input features for a single passenger."""
    Pclass: int = Field(..., ge=1, le=3, example=3,
                        description="Passenger class (1 = First, 2 = Second, 3 = Third)")
    Sex: int = Field(..., ge=0, le=1, example=1,
                     description="Gender encoded: 0 = female, 1 = male")
    Age: float = Field(..., gt=0, lt=120, example=22.0,
                       description="Age in years")
    SibSp: int = Field(..., ge=0, example=1,
                       description="Number of siblings / spouses aboard")
    Parch: int = Field(..., ge=0, example=0,
                       description="Number of parents / children aboard")
    Fare: float = Field(..., ge=0, example=7.25,
                        description="Passenger fare in British pounds")
    Embarked: int = Field(..., ge=0, le=2, example=2,
                          description="Port of embarkation encoded: 0 = Cherbourg, 1 = Queenstown, 2 = Southampton")


class PredictionResponse(BaseModel):
    survived: int = Field(..., description="Prediction: 1 = survived, 0 = did not survive")
    survived_label: str = Field(..., description="Human-readable prediction label")
    probability: float = Field(..., description="Probability of survival (0.0 – 1.0)")


# ──────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────

@app.get("/", summary="Health check")
def root():
    """Return a simple message confirming the API is running."""
    return {"message": "ML API is running"}


@app.post("/predict", response_model=PredictionResponse, summary="Predict survival")
def predict(passenger: PassengerFeatures):
    """
    Accept passenger features, run the trained model,
    and return the survival prediction with probability.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please train the model first by running train.py.",
        )

    # Build feature DataFrame in the exact column order used during training
    feature_order = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    input_data = pd.DataFrame([passenger.dict()])[feature_order]

    # Predict class and probability
    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][1])  # probability of class 1

    label = "Survived" if prediction == 1 else "Did not survive"

    return PredictionResponse(
        survived=prediction,
        survived_label=label,
        probability=round(probability, 4),
    )


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=1912, reload=True)
