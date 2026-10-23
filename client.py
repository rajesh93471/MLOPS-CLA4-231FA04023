import json
import requests

BASE_URL = "http://127.0.0.1:8000"
PASSENGER = {
    "Pclass": 3, "Sex": "male", "Age": 22, "SibSp": 1,
    "Parch": 0, "Fare": 7.25, "Embarked": "S"
}

def main():
    health = requests.get(f"{BASE_URL}/health", timeout=10)
    print("GET /health:", health.status_code, health.json())
    response = requests.post(f"{BASE_URL}/predict", json=PASSENGER, timeout=10)
    print("POST /predict:", response.status_code, response.json())
    response.raise_for_status()

if __name__ == "__main__":
    main()
