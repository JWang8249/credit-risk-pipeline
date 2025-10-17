from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import traceback

app = FastAPI(
    title="Credit Risk Prediction API",
    description="Predict credit default risk from customer financial data.",
    version="1.0.0"
)

# Load model and scaler
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Define input schema
class CreditInput(BaseModel):
    LIMIT_BAL: float = 0
    SEX: int = 0
    EDUCATION: int = 0
    MARRIAGE: int = 0
    AGE: int = 0
    PAY_0: int = 0
    PAY_2: int = 0
    PAY_3: int = 0
    PAY_4: int = 0
    PAY_5: int = 0
    PAY_6: int = 0
    BILL_AMT1: float = 0
    BILL_AMT2: float = 0
    BILL_AMT3: float = 0
    BILL_AMT4: float = 0
    BILL_AMT5: float = 0
    BILL_AMT6: float = 0
    PAY_AMT1: float = 0
    PAY_AMT2: float = 0
    PAY_AMT3: float = 0
    PAY_AMT4: float = 0
    PAY_AMT5: float = 0
    PAY_AMT6: float = 0


@app.get("/")
def root():
    return {"message": "✅ Credit Risk API is running! Visit /docs for testing."}


@app.post("/predict")
def predict(data: CreditInput):
    try:
        # Convert input to DataFrame
        X = pd.DataFrame([data.dict()])

        # Align with training columns
        X = X.reindex(columns=scaler.feature_names_in_, fill_value=0)

        # Scale features
        X_scaled = scaler.transform(X)

        # Predict
        prediction = model.predict(X_scaled)[0]
        risk = "High Risk" if prediction == 1 else "Low Risk"

        return {"prediction": int(prediction), "risk": risk}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
