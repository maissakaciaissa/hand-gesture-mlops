import sys
from pathlib import Path

# allow importing from src/
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from predict import GesturePredictor

from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io


app = FastAPI(title="Hand Gesture Recognition API")

predictor = GesturePredictor(model_path="src/output/model.pth")


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    result = predictor.predict(image)
    return result