import sys
from pathlib import Path

# allow importing from src/
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from predict import GesturePredictor

from fastapi import FastAPI, HTTPException, UploadFile, File
from PIL import Image, UnidentifiedImageError
import io


app = FastAPI(title="Hand Gesture Recognition API")

predictor = GesturePredictor(model_path="src/output/model.pth")

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg"}
MAX_FILE_SIZE_MB = 5

@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Check content type
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Allowed: PNG, JPEG.",
        )
    # 2. Read and check file size
    image_bytes = await file.read()
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size_mb:.1f}MB). Max allowed: {MAX_FILE_SIZE_MB}MB.",
        )

    # 3. Try to decode as an actual image
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.verify()  # checks it's a valid, non-corrupted image
        image = Image.open(io.BytesIO(image_bytes))  # re-open after verify (verify closes it)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not process the uploaded image.")
     # 4. Run prediction, catch unexpected model errors too
    try:
        result = predictor.predict(image)
    except Exception:
        raise HTTPException(status_code=500, detail="Prediction failed unexpectedly.")

    return result