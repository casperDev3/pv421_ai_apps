import cv2
import numpy as np
import time
import os
from pathlib import Path
import face_recognition
import traceback

from face_recognition import face_distance, face_locations
from ultralytics import YOLO
import threading

class FrameGrabber(threading.Thread):
    def __init__(self, scr=0, width=1920, height=1080):
        super().__init__()
        self.capture = cv2.VideoCapture(scr)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.lock = threading.Lock()
        self.last_frame = None
        self.running = True
        if not self.capture.isOpened():
            raise RuntimeError("Could not open video source")

    def run(self):
        while self.running:
            ret, frame = self.capture.read()
            if not ret:
                # print("Failed to grab frame")
                continue
            with self.lock:
                self.last_frame = frame
            # time.sleep(0.01)  # Sleep briefly to reduce CPU usage

    def read(self):
        with self.lock:
            frame_copy = self.last_frame.copy() if self.last_frame is not None else None
        return frame_copy

    def stop(self):
        self.running = False
        self.capture.release() # Release the video capture when stopping
        cv2.destroyAllWindows() # Close all OpenCV windows

class FaceRecognitionSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.face_people_boxes = []

        print("Завантаження моделі YOLOv8n...")
        self.person_detector = YOLO("yolov8n.pt")
        self.person_detector.to("cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu")

        self.frame_count = 0
        self.lock = threading.Lock()

    def load_known_faces(self, photo_folder="photos"):
        print("Завантаження відомих облич...")
        if not os.path.exists(photo_folder):
            print(f"Папка '{photo_folder}' не знайдена. Створіть папку та додайте фотографії.")
            return False

        person_folders = [f for f in Path(photo_folder).interdir() if f.is_dir()]

        if not person_folders:
            print(f"У папці '{photo_folder}' немає підпапок з фотографіями.")
            return False

        total_photos = 0
        for person_folder in person_folders:
            person_name = person_folder.name
            photo_files = list(person_folder.glob("*.*"))
            if not photo_files:
                print(f"У папці '{person_folder}' немає фотографій.")
                continue

            for photo_file in photo_files:
                try:
                    image = face_recognition.load_image_file(photo_file)
                    enc = face_recognition.face_encodings(image)
                    if len(enc):
                        continue
                    self.known_face_encodings.append(enc[0])
                    self.known_face_names.append(person_name)
                    total_photos += 1
                except Exception as e:
                    print(f"Помилка при обробці фотографії '{photo_file}': {e}")
                    continue
        print(f"Завантажено {total_photos} фотографій для {len(self.known_face_names)} осіб.")
        return total_photos > 0

    def process_frame(self, frame, face_interval=3, scale_factor=0.25):
        people_boxes = []
        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        results = self.person_detector(small_frame, classes=[0], conf=0.5, verbose=False)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1, x2, y2 = int(x1 / scale_factor), int(y1 / scale_factor), int(x2 / scale_factor), int(y2 / scale_factor)
                people_boxes.append((x1, y1, x2, y2))

        if self.frame_count % face_interval == 0:
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_recognitions = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_recognitions)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"
                confidence = 0.0
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = (1 - face_distances[best_match_index]) * 100
                face_names.append((name, confidence))

            face_locations =[
                (int(top / scale_factor), int(right / scale_factor), int(bottom / scale_factor), int(left / scale_factor))
                for (top, right, bottom, left) in face_recognitions
            ]

            with self.lock:
                self.face_locations = face_locations
                self.face_names = face_names

        with self.lock:
            self.face_people_boxes = people_boxes

        self.frame_count += 1



if __name__ == "__main__":
    print("Hello World!")
    face_recognition_system = FaceRecognitionSystem()
