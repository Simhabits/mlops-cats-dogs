from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import time

from src.utils.model_loader import load_model
from src.utils.preprocess import preprocess_image

MODEL_PATH = "output/model.pt"
CLASS_NAMES = ["cat", "dog"]

app = FastAPI(
    title="Cats vs Dogs Inference API",
    version="1.0"
)

model = load_model(MODEL_PATH)

request_count = 0

# -----------------------
# Health Check
# -----------------------
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True
    }

# -----------------------
# Prediction Endpoint
# -----------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global request_count
    start_time = time.time()

    image = Image.open(file.file).convert("RGB")
    input_tensor = preprocess_image(image)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0].tolist()
        predicted_class = CLASS_NAMES[torch.argmax(outputs).item()]

    latency = round(time.time() - start_time, 4)
    request_count += 1

    return {
        "prediction": predicted_class,
        "probabilities": {
            "cat": round(probs[0], 4),
            "dog": round(probs[1], 4)
        },
        "latency_sec": latency,
        "request_count": request_count
    }
