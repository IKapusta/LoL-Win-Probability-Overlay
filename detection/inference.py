import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from paths import DETECTION_DIR

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


classes = [str(i) for i in range(1, 19)]  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=len(classes))
model.load_state_dict(torch.load(DETECTION_DIR / "digit_classifier.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((68, 56)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def classify_image(image_path):
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor) 
        probabilities = torch.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, dim=1)
        if confidence < 0.8:
            prediction = torch.argmax(output, dim=1).item()

            print("LOW CONFIDENCE PREDICTION:", classes[prediction])
            return 0
        prediction = torch.argmax(output, dim=1).item()

    return classes[prediction]


if __name__ == "__main__":
    img_path = "./scoreboards/upscaled_area_0.png"
    predicted_class = classify_image(img_path)
    print(f"Predicted class: {predicted_class}")