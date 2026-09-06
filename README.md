# Hand Gesture Recognition — MLOps Project

A computer vision project that trains a CNN to classify hand gestures from the [LeapGestRecog](https://www.kaggle.com/datasets/gti-upm/leapgestrecog) dataset, serves it through a FastAPI endpoint, and wraps the whole thing in a working MLOps loop: experiment tracking, containerization, and CI/CD.

The focus wasn't just "train a model" — it's the full pipeline: data → training → diagnosis of overfitting → serving → packaging → automated testing.

## Features

- Custom CNN (PyTorch) trained on 20,000 grayscale hand images, 10 gesture classes
- Subject-based train/val split (not random) to properly test generalization to unseen people
- Data augmentation, batch normalization, weight decay, and LR scheduling to control overfitting
- FastAPI backend serving predictions over HTTP
- Dockerized application
- Experiment tracking and model registry with MLflow
- Automated testing (pytest) and CI/CD (GitHub Actions): tests → Docker build → container smoke test, on every push

## Architecture

```
                Kaggle (GPU training)
                        |
                        v
              train.py --> model.pth
                        |
                        v
                  MLflow tracking
              (params, metrics, model registry)
                        |
                        v
                 src/predict.py
              (GesturePredictor class)
                        |
                        v
                  FastAPI (api/main.py)
                     /predict
                        |
                        v
                  Docker container
                        |
                        v
              GitHub Actions CI/CD
        pytest --> docker build --> smoke test
```

## Project Structure

```
hand-gesture-mlops/
│
├── src/
│   ├── dataset.py        # LeapGestRecog loader, subject-based split
│   ├── model.py           # CNN architecture (conv + batchnorm + pooling)
│   ├── train.py            # Training loop, MLflow logging, checkpointing
│   ├── predict.py          # GesturePredictor: loads model, runs inference
│   └── output/
│       └── model.pth       # Trained weights + class list
│
├── api/
│   └── main.py             # FastAPI app, /predict endpoint
│
├── tests/
│   └── test_predict.py     # Unit tests for GesturePredictor + prediction correctness
│
├── data/
│   └── *.png                # Sample images for local testing / CI smoke test
│
├── notebooks/
│   └── hand-gesture-rec.ipynb  # Kaggle training notebook
│
├── .github/workflows/
│   └── docker-build.yml    # CI: pytest -> docker build -> smoke test
│
├── Dockerfile
├── .dockerignore
├── pyproject.toml / uv.lock # Dependency management (uv)
└── README.md
```

## Dataset

[LeapGestRecog](https://www.kaggle.com/datasets/gti-upm/leapgestrecog) — ~20,000 grayscale hand images from 10 subjects, each performing 10 gestures (palm, fist, ok, thumb, index, c, down, l, palm_moved, fist_moved).

**Split strategy:** training on subjects 00–07, validating on subjects 08–09 — entire people held out, not random images. This tests whether the model generalizes to a new person's hand rather than memorizing individual subjects' silhouettes.

## Training & Results

- Architecture: 3-layer CNN with batch normalization, dropout, ~64×64 grayscale input
- Optimizer: Adam with weight decay, `ReduceLROnPlateau` scheduler
- Augmentation: random rotation + affine (translate/scale) on training data only

Iterating on the model surfaced a real overfitting problem (near-perfect train accuracy, unstable/lower validation accuracy) caused by the model learning subject-specific hand shapes rather than general gesture shapes. Fixed progressively with augmentation, batchnorm, weight decay, and LR scheduling.

**Best result:** ~97% validation accuracy on fully unseen subjects.

Full metrics and training curves are tracked in MLflow (`mlruns/`, `mlflow.db`).

## Running Locally

**Install dependencies:**
```bash
uv sync
```

**Run inference on a single image:**
```bash
uv run python src/predict.py path/to/image.png
```

**Run the API:**
```bash
uv run uvicorn api.main:app --reload
```
Then open `http://127.0.0.1:8000/docs` to try the `/predict` endpoint interactively.

**Run tests:**
```bash
uv run pytest
```

## Running with Docker

```bash
docker build -t hand-gesture-api .
docker run -p 8000:8000 hand-gesture-api
```
API available at `http://127.0.0.1:8000`.

## Experiment Tracking (MLflow)

Training runs (params, metrics, model artifacts) are logged with MLflow. To view:
```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root file:./mlruns
```
Then open `http://127.0.0.1:5000`.

The best model is registered in the MLflow Model Registry as `hand-gesture-cnn`.

## CI/CD

Every push to `main` triggers a GitHub Actions workflow:
1. **Test** — installs dependencies, runs `pytest`
2. **Build** — builds the Docker image
3. **Smoke test** — runs the container, sends a real image to `/predict`, verifies a valid response

If tests fail, the Docker build is skipped entirely — fast feedback before the more expensive build step runs.

## Tech Stack

PyTorch · OpenCV · FastAPI · Docker · MLflow · GitHub Actions · pytest · uv