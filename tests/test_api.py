import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_predict_valid_image():
    with open("data/frame_09_01_0025.png", "rb") as f:
        response = client.post(
            "/predict",
            files={"file": ("frame_09_01_0025.png", f, "image/png")},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "palm"


def test_predict_rejects_wrong_content_type():
    response = client.post(
        "/predict",
        files={"file": ("fake.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400


def test_predict_rejects_corrupted_image():
    response = client.post(
        "/predict",
        files={"file": ("fake.png", b"this is not really a png", "image/png")},
    )
    assert response.status_code == 400