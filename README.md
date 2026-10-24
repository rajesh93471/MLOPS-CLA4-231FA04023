# Titanic Model Serving: Online, Batch, Streaming, and FastAPI

This project implements the assignment tasks using the **Kaggle Titanic `train.csv` dataset**.

## Project files

- `model.py` — preprocesses the data, creates a stratified train/test split, trains a Random Forest model, evaluates accuracy, and saves the fitted pipeline.
- `online_inference.py` — predicts one unseen hold-out record.
- `batch_inference.py` — predicts 20 unseen hold-out records and saves `batch_predictions.csv`.
- `streaming_inference.py` — processes 10 unseen records one at a time, with a one-second delay.
- `app.py` — provides FastAPI endpoints `GET /health` and `POST /predict`, with Pydantic validation.
- `client.py` — Python HTTP client for the API.
- `test_model.py` — pytest tests for valid and invalid requests.
- `generate_report.py` — creates `assignment_report.pdf` with the four required observations.
- `run_all.py` — trains once and runs the inference demos and report generation.
- `capture_outputs.py` — creates evidence images from online, batch, streaming, and FastAPI endpoint results.
- `requirements.txt` — required Python dependencies.

## Dataset

Use only the Titanic `train.csv` file specified by the assignment: [Kaggle Titanic competition data](https://www.kaggle.com/competitions/titanic/data).

The required `data/train.csv` is included in this ZIP. It contains 891 labeled rows with the original Kaggle Titanic training columns. `download_dataset.py` is included as a fallback if you need to retrieve the data again.

## Setup and execution

Use Python 3.10 or newer.
```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate

# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python model.py
python online_inference.py
python batch_inference.py
python streaming_inference.py
python generate_report.py
pytest -q
```

Or run the complete non-API workflow with:
```
python run_all.py
```

The ZIP includes `data/train.csv`, the trained `artifacts/model.pkl` (also copied to the project root as `model.pkl`), generated batch predictions, metrics, and a PDF report.

The included run used a stratified 80/20 split: 712 training rows and 179 hold-out test rows. Measured hold-out accuracy was 81.56%. The same saved pipeline is reused for all inference strategies and API serving. The 20 batch records and 10 streaming records come from the hold-out split, not the rows used to fit the model.

## Run and test FastAPI

After running `python model.py`, start the API:
```
uvicorn app:app --reload
```

Open Swagger UI at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

1. Try `GET /health`.
2. Try `POST /predict` with a valid body:
```json
{
  "Pclass": 3,
  "Sex": "male",
  "Age": 22,
  "SibSp": 1,
  "Parch": 0,
  "Fare": 7.25,
  "Embarked": "S"
}
```

3. Try an invalid body, such as `Pclass: 9`, `Sex: "unknown"`, or a negative age. It should be rejected with HTTP 422.
4. Run `python client.py` to test the endpoint from a Python client.

## Outputs

- `artifacts/model.pkl` — saved preprocessing and classifier pipeline.
- `artifacts/metrics.json` — dataset split counts and measured hold-out accuracy.
- `artifacts/holdout_unseen_records.csv` — hold-out records reserved for demonstrations.
- `batch_predictions.csv` and `outputs/` — inference results.
- `assignment_report.pdf` — observation report.
- `screenshots/` — evidence images showing program output and FastAPI endpoint responses. The FastAPI evidence was captured with FastAPI TestClient, not from a browser view of Swagger UI. If your instructor specifically requires a browser screenshot, open `/docs`, repeat the requests, and capture the page.

## Required observations

1. **Online vs. batch inference:** Online inference predicts one record at a time and returns an immediate result. Batch inference predicts a group of records together.
2. **Streaming vs. batch inference:** Streaming processes each incoming record sequentially, while batch inference processes a collected group together.
3. **Why FastAPI?** It provides REST endpoints, request validation, interactive Swagger documentation, and a convenient way for other applications to call the model.
4. **Why reuse one trained model?** The same fitted preprocessing and classifier keep predictions consistent across online, batch, streaming, and API serving.
