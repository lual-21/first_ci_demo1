import json
import sys

# Minimum required accuracy
MIN_ACCURACY = 0.85

# Load model evaluation metrics
with open("metrics.json", "r") as file:
    metrics = json.load(file)

accuracy = metrics["accuracy"]

print(f"Required accuracy: {MIN_ACCURACY:.2f}")
print(f"Actual accuracy:   {accuracy:.2f}")

# Quality gate
if accuracy < MIN_ACCURACY:
    print("QUALITY GATE FAILED!")
    print("Model accuracy is below the required threshold.")
    sys.exit(1)

print("QUALITY GATE PASSED!")
sys.exit(0)
