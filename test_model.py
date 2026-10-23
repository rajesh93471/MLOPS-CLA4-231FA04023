from pathlib import Path
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from model import FEATURES, train_and_save, load_model
import app as api

@pytest.fixture(scope="session", autouse=True)
def ensure_model():
    if not Path("artifacts/model.pkl").exists():
        train_and_save()

def test_model_artifact_exists():
    assert Path("artifacts/model.pkl").exists()

def test_model_predicts_binary_class():
    model = load_model()
    sample = pd.DataFrame([{
        "Pclass": 3, "Age": 22, "SibSp": 1, "Parch": 0,
        "Fare": 7.25, "Sex": "male", "Embarked": "S"
    }])[FEATURES]
    prediction = int(model.predict(sample)[0])
    assert prediction in (0, 1)

def test_health_endpoint():
    client = TestClient(api.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_valid_request():
    client = TestClient(api.app)
    response = client.post("/predict", json={
        "Pclass": 3, "Sex": "male", "Age": 22, "SibSp": 1,
        "Parch": 0, "Fare": 7.25, "Embarked": "S"
    })
    assert response.status_code == 200
    assert response.json()["prediction"] in (0, 1)

def test_predict_invalid_request_rejected():
    client = TestClient(api.app)
    response = client.post("/predict", json={
        "Pclass": 9, "Sex": "unknown", "Age": -5, "SibSp": -1,
        "Parch": 0, "Fare": -3, "Embarked": "X"
    })
    assert response.status_code == 422
