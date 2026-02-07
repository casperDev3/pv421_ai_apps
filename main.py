import torch # для роботи з нейронними мережами
import torchvision.transforms as transforms
import torchvision.models as models
import cv2 # малювання графіки
import requests
import time
# import numpy as np


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

def draw_text_with_background(frame, text, position, font_scale=0.8, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y - text_height - baseline), (x + text_width, y + baseline), (0, 0, 0), -1) # малювання чорного прямокутника для фону тексту
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame) # накладання прямокутника на зображення

    cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness) # малювання тексту поверх прямокутника

def run_real_time_detection(camera_id=0, confidence_threshold=0.3):
    cap = cv2.VideoCapture(camera_id) # відкриття відеопотоку (0 для веб-камери)

    if not cap.isOpened():
        print("Помилка: Не вдалося відкрити відеопотік.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # встановлення ширини кадру
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # встановлення висоти кад

    print("Починаємо обробку відеопотоку. Натисніть 'q' для виходу.")

    fps_start_time = time.time()
    fps_counter = 0
    fps = 0

    colors = [
        (0, 255, 0),  # Зелений
        (255, 0, 0),  # Синій
        (0, 0, 255),  # Червоний
        (255, 255, 0),  # Блакитний
        (255, 0, 255),  # Пурпурний
        (0, 255, 255),  # Жовтий
    ]

    while True:
        ret, frame = cap.read() # зчитування кадру з відеопотоку
        if not ret:
            print("Помилка: Не вдалося зчитати кадр.")
            break

        try:
            results = yolo_model(frame)  # передбачення об'єктів на кадрі за допомогою YOLOv5
            detections = results.pandas().xyxy[0] # отримання координат та класів виявлених об'єктів

            for idx, detection in detections.iterrows():
                confidence = detection['confidence']
                if confidence < confidence_threshold:
                    continue  # пропуск об'єктів з низькою ймовірністю

                x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
                class_name = detection['name']
                color = colors[idx % len(colors)]  # вибір кольору для кожного класу

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # малювання прямокутника навколо виявленого об'єкта

                label = f"{class_name} - {round(confidence * 100, 2)}%"  # створення тексту з назвою класу та ймовірністю
                draw_text_with_background(frame, label, (x1, y1 - 10))  # малювання тексту з фоном над прямокутником
        except Exception as e:
            print(f"Помилка при обробці кадру: {e}")

        fps_counter +=  1
        if time.time() - fps_start_time >= 1.0:  # оновлення FPS кожну секунду
            fps = fps_counter
            fps_counter = 0
            fps_start_time = time.time()

        # Відображення FPS накадрі
        draw_text_with_background(frame, f"FPS: {fps}", (10, 30))  # малювання FPS в верхньому лівому куті
        cv2.imshow('Real-Time Object Detection', frame)  # відображення кадру з виявленими об'єктами

        if cv2.waitKey(1) & 0xFF == ord('q'):  # вихід з циклу при натисканні 'q'
            break

    cap.release()
    cv2.destroyAllWindows()  # закриття всіх вікон OpenCV
    print("Завершено обробку відеопотоку.")

def main():
    try:
        run_real_time_detection(camera_id=1, confidence_threshold=0.2)
    except KeyboardInterrupt:
        print("Завершення роботи за запитом користувача.")
    except Exception as e:
        print(f"Помилка: {e}")

if __name__ == "__main__":
    main()