import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from predict import GesturePredictor
from PIL import Image

MODEL_PATH = "src/output/model.pth"
SAMPLE_IMAGE = "data/frame_09_01_0025.png"  # known ground truth: palm


def test_predictor_loads():
    predictor = GesturePredictor(model_path=MODEL_PATH)
    assert predictor.model is not None
    assert len(predictor.classes) == 10


def test_prediction_shape():
    predictor = GesturePredictor(model_path=MODEL_PATH)
    image = Image.open(SAMPLE_IMAGE)
    result = predictor.predict(image)

    assert "label" in result
    assert "confidence" in result
    assert isinstance(result["confidence"], float)


def test_prediction_correctness():
    predictor = GesturePredictor(model_path=MODEL_PATH)
    image = Image.open(SAMPLE_IMAGE)
    result = predictor.predict(image)           

    assert result["label"] == "palm"
    assert result["confidence"] > 0.5