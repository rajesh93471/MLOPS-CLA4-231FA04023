from pathlib import Path
import json
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from download_dataset import main as ensure_dataset

DATA_PATH = Path("data/train.csv")
ARTIFACTS = Path("artifacts")
MODEL_PATH = ARTIFACTS / "model.pkl"
METRICS_PATH = ARTIFACTS / "metrics.json"
NUMERIC = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CATEGORICAL = ["Sex", "Embarked"]
FEATURES = NUMERIC + CATEGORICAL

def build_pipeline():
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    prep = ColumnTransformer([
        ("numeric", numeric, NUMERIC),
        ("categorical", categorical, CATEGORICAL),
    ])
    return Pipeline([
        ("preprocessing", prep),
        ("classifier", RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced",
            min_samples_leaf=2
        )),
    ])

def load_dataset():
    if not DATA_PATH.exists():
        ensure_dataset()
    df = pd.read_csv(DATA_PATH)
    required = set(FEATURES + ["Survived"])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"train.csv is missing columns: {sorted(missing)}")
    df = df.dropna(subset=["Survived"]).copy()
    return df

def train_and_save():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    X = df[FEATURES]
    y = df["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    accuracy = float(accuracy_score(y_test, predictions))
    joblib.dump(pipeline, MODEL_PATH)
    metrics = {
        "model_name": "Random Forest Classifier",
        "dataset": "Kaggle Titanic - train.csv",
        "dataset_rows": int(len(df)),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "accuracy": accuracy,
        "accuracy_percent": round(accuracy * 100, 2),
        "random_state": 42,
        "features": FEATURES,
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    # Hold-out rows are never used to fit the model and are used for inference demos.
    holdout = X_test.copy()
    holdout.insert(0, "PassengerId", df.loc[X_test.index, "PassengerId"].values if "PassengerId" in df.columns else X_test.index)
    holdout["ActualSurvived"] = y_test.values
    holdout.to_csv(ARTIFACTS / "holdout_unseen_records.csv", index=False)
    print(f"Model saved: {MODEL_PATH}")
    print(f"Hold-out accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    return pipeline, holdout, metrics

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run `python model.py` first.")
    return joblib.load(MODEL_PATH)

if __name__ == "__main__":
    train_and_save()
