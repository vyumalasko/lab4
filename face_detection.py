import cv2
import numpy as np

import urllib.request

yunet_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
model_path = "face_detection_yunet_2023mar.onnx"

urllib.request.urlretrieve(yunet_url, model_path)

print("Модель YuNET успешно скачана!")

model_path = "face_detection_yunet_2023mar.onnx"
detector = cv2.FaceDetectorYN.create(model_path, "", (320, 320))

cap = cv2.VideoCapture(0)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
detector.setInputSize((frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Ошибка получения кадра")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = detector.detect(frame_rgb)

    if faces[1] is not None:
        for face in faces[1]:
            x, y, w, h = map(int, face[:4])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()