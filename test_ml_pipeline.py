import json
import os
import joblib

def test_model_exists():
    assert os.path.exists("student_result_model.pkl")

def test_metrics_exist():
    assert os.path.exists("metrics.json")

def test_accuracy_is_valid():
    with open("metrics.json", "r") as file:
        metrics = json.load(file)

    assert 0.0 <= metrics["accuracy"] <= 1.0

def test_model_prediction():
    model = joblib.load("student_result_model.pkl")

    prediction = model.predict([[80, 4, 75]])

    assert prediction[0] in [0, 1]

if __name__ == "__main__":
    test_model_exists()
    test_metrics_exist()
    test_accuracy_is_valid()
    test_model_prediction()

    print("All ML pipeline tests passed!")
