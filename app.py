from pathlib import Path
from typing import Literal
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

MODEL_PATH = Path("artifacts/model.pkl")
FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked"]
app = FastAPI(title="Titanic Survival Prediction API", version="1.0.0",
              description="Serves the trained Kaggle Titanic classification pipeline.")

class Passenger(BaseModel):
    model_config = ConfigDict(extra="forbid")
    Pclass: int = Field(ge=1, le=3, description="Ticket class: 1, 2, or 3")
    Sex: Literal["male", "female"]
    Age: float | None = Field(default=None, ge=0, le=100)
    SibSp: int = Field(ge=0, le=20)
    Parch: int = Field(ge=0, le=20)
    Fare: float = Field(ge=0)
    Embarked: Literal["C", "Q", "S"] | None = "S"

def get_model():
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model artifact is missing. Run python model.py first.")
    return joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}

@app.post("/predict")
def predict(passenger: Passenger):
    model = get_model()
    data = pd.DataFrame([passenger.model_dump()])[FEATURES]
    result = int(model.predict(data)[0])
    return {"prediction": result, "label": "Survived" if result == 1 else "Did not survive"}
