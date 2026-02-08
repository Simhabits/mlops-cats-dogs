import requests

TEST_SAMPLES = {
    "dog.4010.jpg": "dog",
    "dog.4019.jpg": "dog",
    "cat.4029.jpg": "cat"


}

correct = 0
total = 0

for image, true_label in TEST_SAMPLES.items():
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": open(f"tests/assets/{image}", "rb")}
    ).json()

    predicted = response["prediction"]

    print(f"{image} → Predicted: {predicted}, Actual: {true_label}")

    if predicted == true_label:
        correct += 1

    total += 1

accuracy = correct / total
print(f"\nPost-deployment accuracy: {accuracy:.2f}")
