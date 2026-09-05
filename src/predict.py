import torch
from PIL import Image
from torchvision import transforms

from model import GestureCNN

MODEL_PATH = "output/model.pth"
IMG_SIZE = 64


class GesturePredictor:
    def __init__(self, model_path=MODEL_PATH, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(model_path, map_location=self.device)
        self.classes = checkpoint["classes"]

        self.model = GestureCNN(num_classes=len(self.classes))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ])

    def predict(self, image: Image.Image):
        image = image.convert("L")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = probs.max(1)

        label = self.classes[predicted_idx.item()]
        return {
            "label": label,
            "confidence": confidence.item(),
        }


if __name__ == "__main__":
    import sys

    predictor = GesturePredictor()
    image_path = sys.argv[1]
    image = Image.open(image_path)

    result = predictor.predict(image)
    print(result)