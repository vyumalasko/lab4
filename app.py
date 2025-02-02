import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os
from deepface import DeepFace

df = pd.read_csv("staff_photo.csv")

def find_closest_face(uploaded_file):
    print("Загрузка изображения...")
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    print("Изображение загружено, начинаем поиск...")
    min_distance = float("inf")
    closest_match = None

    for _, row in df.iterrows():
        face_path = os.path.join("hse_faces_miem", row["filename"])
        if os.path.exists(face_path):
            try:
                print(f"Сравнение с {row['filename']}...")
                result = DeepFace.verify(img, face_path, model_name="Facenet", enforce_detection=False)
                distance = result["distance"]
                print(f"Результат: {result}")
                if distance < min_distance:
                    min_distance = distance
                    closest_match = row
            except Exception as e:
                print(f"Ошибка при сравнении с {row['filename']}: {e}")
                
    return closest_match, min_distance


# Интерфейс Streamlit
st.title("🔍 Поиск лица в базе HSE MIEM")

uploaded_file = st.file_uploader("📤 Загрузите изображение", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Загруженное изображение", use_container_width=True)

    if st.button("🔍 Найти похожее лицо"):
        with st.spinner("Идет поиск..."):
            match, distance = find_closest_face(uploaded_file)

            if match is not None:
                st.success(f"👤 Найдено лицо: {match['name']} (расстояние: {distance:.2f})")
                face_path = os.path.join("hse_faces_miem", match["filename"])
                st.image(face_path, caption=match["name"], use_container_width=True)
            else:
                st.error("❌ Лицо не найдено")
