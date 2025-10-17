import pytest
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

MODEL_PATH = "models/model.pkl"
SCALER_PATH = "models/scaler.pkl"

@pytest.mark.order(1)
def test_model_loads_successfully():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    assert isinstance(model, LogisticRegression)
    assert hasattr(scaler, "transform")

@pytest.mark.order(2)
def test_model_can_predict():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    n_features = len(scaler.feature_names_in_)  
    X_dummy = np.random.rand(1, n_features)
    X_scaled = scaler.transform(X_dummy)
    prediction = model.predict(X_scaled)
    assert prediction.shape == (1,)
    assert prediction[0] in [0, 1]

@pytest.mark.order(3)
def test_model_reproducibility():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    n_features = len(scaler.feature_names_in_)
    X = np.random.rand(1, n_features)
    X_scaled = scaler.transform(X)
    pred1 = model.predict(X_scaled)
    pred2 = model.predict(X_scaled)
    assert (pred1 == pred2).all()
