import cv2
import numpy as np
from deepface import DeepFace  # Библиотека для сравнения лиц

# Путь к модели YuNET
model_path = "face_detection_yunet_2023mar.onnx"
detector = cv2.FaceDetectorYN.create(model_path, "", (320, 320))


selfie_path = "selfie.jpg"
selfie_embedding = DeepFace.represent(img_path=selfie_path, model_name="Facenet", enforce_detection=False)

# Подключаем камеру
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
            x, y, w, h = map(int, face[:4])  # Координаты лица
            face_img = frame_rgb[y:y+h, x:x+w]  # Вырезаем лицо из кадра
            
            try:
                # Извлекаем эмбеддинг найденного лица
                face_embedding = DeepFace.represent(img_path=face_img, model_name="Facenet", enforce_detection=False)

                # Сравниваем с эталоном
                distance = np.linalg.norm(np.array(face_embedding) - np.array(selfie_embedding))

                # Если расстояние маленькое → это ты!
                if distance < 10:  
                    color = (0, 255, 0)  # Зелёный (твоё лицо)
                else:
                    color = (0, 0, 255)  # Красный (чужие лица)

            except:
                color = (0, 0, 255)  # Если ошибка, считаем лицо чужим

            # Рисуем прямоугольник вокруг лица
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    cv2.imshow("Face Verification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
