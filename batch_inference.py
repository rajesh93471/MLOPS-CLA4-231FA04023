from pathlib import Path
import pandas as pd
from model import FEATURES, load_model, train_and_save

def main():
    if not Path("artifacts/model.pkl").exists():
        train_and_save()
    model = load_model()
    holdout = pd.read_csv("artifacts/holdout_unseen_records.csv")
    if len(holdout) < 20:
        raise ValueError("At least 20 hold-out records are needed.")
    batch = holdout.iloc[1:21].copy()
    batch["PredictedSurvived"] = model.predict(batch[FEATURES]).astype(int)
    batch["PredictedLabel"] = batch["PredictedSurvived"].map({0: "Did not survive", 1: "Survived"})
    batch.to_csv("batch_predictions.csv", index=False)
    Path("outputs").mkdir(exist_ok=True)
    batch.to_csv("outputs/batch_predictions.csv", index=False)
    print(f"BATCH INFERENCE: predicted {len(batch)} unseen hold-out records")
    print(batch[["PassengerId", "PredictedSurvived", "PredictedLabel"]].to_string(index=False))

if __name__ == "__main__":
    main()
