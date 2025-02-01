import cv2
import numpy as np

import urllib.request

# Ссылка на YuNET ONNX модель
yunet_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
model_path = "face_detection_yunet_2023mar.onnx"

# Скачиваем файл
urllib.request.urlretrieve(yunet_url, model_path)

print("Модель YuNET успешно скачана!")

model_path = "face_detection_yunet_2023mar.onnx"
detector = cv2.FaceDetectorYN.create(model_path, "", (320, 320))


# Параметры видеопотока
cap = cv2.VideoCapture(0)  # 0 - основная веб-камера

# Настраиваем YuNET под разрешение видео
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
detector.setInputSize((frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Ошибка получения кадра")
        break

    # Конвертируем кадр в формат RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Обнаружение лиц
    faces = detector.detect(frame_rgb)

    if faces[1] is not None:
        for face in faces[1]:
            x, y, w, h = map(int, face[:4])  # Координаты лица
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Выводим изображение
    cv2.imshow("Face Detection", frame)

    # Нажатие "q" - выход
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()