from pathlib import Path
import urllib.request

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
TRAIN_PATH = DATA_DIR / "train.csv"
URLS = [
    "https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/train.csv",
    "https://raw.githubusercontent.com/pcsanwald/kaggle-titanic/master/train.csv",
]

def main():
    if TRAIN_PATH.exists():
        print(f"Dataset already exists: {TRAIN_PATH}")
        return
    errors = []
    for url in URLS:
        try:
            print(f"Downloading Kaggle Titanic train.csv from public mirror: {url}")
            urllib.request.urlretrieve(url, TRAIN_PATH)
            # Basic validation prevents saving an unrelated file.
            import pandas as pd
            df = pd.read_csv(TRAIN_PATH)
            required = {"Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"}
            if len(df) < 100 or not required.issubset(df.columns):
                TRAIN_PATH.unlink(missing_ok=True)
                raise ValueError("Downloaded CSV did not match the Titanic training schema.")
            print(f"Saved {len(df)} rows to {TRAIN_PATH}")
            return
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            TRAIN_PATH.unlink(missing_ok=True)
    raise RuntimeError("Could not download train.csv. Download it manually from https://www.kaggle.com/competitions/titanic/data and save it as data/train.csv.\n" + "\n".join(errors))

if __name__ == "__main__":
    main()
