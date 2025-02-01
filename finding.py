import os
import numpy as np
from deepface import DeepFace
import faiss

# Папка с изображениями
image_folder = 'hse_faces_miem'

# Собираем пути ко всем изображениям в папке
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Генерация эмбеддингов для каждого изображения
embeddings = []
for image_path in image_paths:
    embedding = DeepFace.represent(img_path=image_path, model_name="Facenet", enforce_detection=False)[0]['embedding']
    embeddings.append(np.array(embedding))

# Конвертируем список эмбеддингов в numpy массив
embeddings = np.array(embeddings)

# Создаём индекс для поиска
index = faiss.IndexFlatL2(embeddings.shape[1])  # Индекс на основе L2 нормы
index.add(embeddings)  # Добавляем эмбеддинги в индекс

# Теперь ты можешь использовать этот индекс для поиска похожих лиц
# Например, поиск похожих лиц по запросу
query_image_path = 'selfie.jpg'  # Путь к изображению, с которым будем сравнивать
query_embedding = DeepFace.represent(img_path=query_image_path, model_name="Facenet", enforce_detection=False)[0]['embedding']
query_embedding = np.array(query_embedding).reshape(1, -1)

# Поиск наиболее похожих изображений
k = 3  # Количество наиболее похожих изображений
distances, indices = index.search(query_embedding, k)

import pandas as pd

# Загружаем CSV-файл с ФИО
csv_path = "staff_photo.csv"
df = pd.read_csv(csv_path)

# Преобразуем в словарь {filename: name}
filename_to_name = dict(zip(df["filename"], df["name"]))

for i, distance in zip(indices[0], distances[0]):
    filename = os.path.basename(image_paths[i])  # Получаем имя файла без пути
    name = filename_to_name.get(filename, "Неизвестно")  # Берём ФИО или пишем "Неизвестно"
    print(f"ФИО: {name}, Файл: {filename}, Расстояние: {distance:.2f}")

