import json
from pathlib import Path
import pandas as pd
from model import FEATURES, load_model, train_and_save

def main():
    if not Path("artifacts/model.pkl").exists():
        train_and_save()
    model = load_model()
    holdout = pd.read_csv("artifacts/holdout_unseen_records.csv")
    row = holdout.iloc[[0]]
    prediction = int(model.predict(row[FEATURES])[0])
    result = {
        "PassengerId": int(row.iloc[0]["PassengerId"]),
        "prediction": prediction,
        "label": "Survived" if prediction == 1 else "Did not survive",
        "actual_label_for_demo_evaluation_only": int(row.iloc[0]["ActualSurvived"]),
    }
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/online_prediction.json").write_text(json.dumps(result, indent=2))
    print("ONLINE INFERENCE")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
