import pandas as pd
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Create a sample student dataset
data = {
    "attendance": [90, 80, 45, 70, 95, 50, 85, 60, 75, 40,
                   92, 65, 88, 55, 78, 30, 96, 72, 82, 48],
    "study_hours": [5, 4, 1, 3, 6, 2, 5, 2, 4, 1,
                    6, 3, 5, 2, 4, 1, 7, 3, 5, 2],
    "previous_marks": [85, 75, 40, 65, 90, 45, 80, 55, 70, 35,
                       95, 60, 88, 50, 76, 25, 98, 68, 83, 42],
    "result": [1, 1, 0, 1, 1, 0, 1, 0, 1, 0,
               1, 1, 1, 0, 1, 0, 1, 1, 1, 0]
}

df = pd.DataFrame(data)

# Save the dataset
df.to_csv("student_results.csv", index=False)

# 2. Separate features and target
X = df[["attendance", "study_hours", "previous_marks"]]
y = df["result"]

# 3. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# 4. Train the machine-learning model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# 5. Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# 6. Save the trained model
joblib.dump(model, "student_result_model.pkl")

# 7. Save evaluation metrics
metrics = {"accuracy": float(accuracy)}

with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print(f"Model accuracy: {accuracy:.2f}")
print("Model training completed.")
