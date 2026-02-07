import torch # для роботи з нейронними мережами
import torchvision.transforms as transforms
import torchvision.models as models
import cv2 # малювання графіки
import requests
import time
import numpy as np
from sympy.stats.rv import probability

print("Завантаження моделі YOLOv5...")
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True) # завантаження моделі YOLOv5s
yolo_model.eval() # встановлення моделі в режим оцінки

print("Завантаження моделі ResNet50...")
model = models.resnet50(pretrained=True) # завантаження моделі ResNet50
model.eval() # встановлення моделі в режим оцінки

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # вибір пристрою (GPU або CPU)
model = model.to(device) # переміщення моделі на вибраний пристрій
yolo_model = yolo_model.to(device) # переміщення моделі YOLOv5 на вибраний пристрій
print(f"Моделі завантажено та готові до роботи на пристрої: {device}")

LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = requests.get(LABELS_URL).json()

# Функція для обробки зображення перед передачею в ResNet50
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # конвертація зображення з BGR в RGB
    img_tensor = transform(frame_rgb).unsqueeze(0).to(device) # обробка зображення та переміщення на пристрій

    with torch.no_grad():
        outputs = model(img_tensor) # передбачення класу зображення

    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    top_prob, top_idx = torch.max(probabilities, 0)

    top_class = labels[top_idx.item()] # отримання назви класу за індексом
    confidence = top_prob.item() # отримання ймовірності передбачення

    return top_class, confidence

if __name__ == "__main__":
    print("Hello, World!")