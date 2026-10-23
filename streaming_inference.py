import time
from pathlib import Path
import pandas as pd
from model import FEATURES, load_model, train_and_save

def main(delay_seconds=1):
    if not Path("artifacts/model.pkl").exists():
        train_and_save()
    model = load_model()
    holdout = pd.read_csv("artifacts/holdout_unseen_records.csv")
    stream = holdout.iloc[21:31]
    if len(stream) < 10:
        raise ValueError("At least 10 hold-out records are needed.")
    print("STREAMING INFERENCE (one record at a time)")
    results = []
    for _, row in stream.iterrows():
        pred = int(model.predict(pd.DataFrame([row[FEATURES].to_dict()]))[0])
        item = {"PassengerId": int(row["PassengerId"]), "PredictedSurvived": pred}
        results.append(item)
        print(f"Incoming PassengerId={item['PassengerId']} -> prediction={pred} ({'Survived' if pred else 'Did not survive'})")
        time.sleep(delay_seconds)
    Path("outputs").mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv("outputs/streaming_predictions.csv", index=False)

if __name__ == "__main__":
    main()
