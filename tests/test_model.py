import os
import joblib
import numpy as np
import pytest

MODEL_PATH = "models/model.pkl"
SCALER_PATH = "models/scaler.pkl"

@pytest.mark.order(1)
def test_model_and_scaler_exist():
    """
    Verify that the trained model and scaler files exist.
    """
    assert os.path.exists(MODEL_PATH), "❌ model.pkl not found — run `make train` first."
    assert os.path.exists(SCALER_PATH), "❌ scaler.pkl not found — run `make train` first."


@pytest.mark.order(2)
def test_model_can_predict():
    """
    Ensure that the model produces a valid numeric prediction.
    """
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Create random dummy input with the same shape as training data (5 features here)
    X_dummy = np.random.rand(1, 5)
    X_scaled = scaler.transform(X_dummy)
    pred = model.predict(X_scaled)

    assert pred.shape == (1,), "❌ Prediction output shape incorrect."
    assert pred[0] in [0, 1], "❌ Model prediction should be 0 or 1."


@pytest.mark.order(3)
def test_model_reproducibility():
    """
    Check that two predictions on identical inputs are the same (model determinism).
    """
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    X = np.random.rand(1, 5)
    X_scaled = scaler.transform(X)
    pred1 = model.predict(X_scaled)
    pred2 = model.predict(X_scaled)
    assert np.array_equal(pred1, pred2), "❌ Model predictions are not reproducible."

