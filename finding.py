import os
import numpy as np
import faiss
import pandas as pd
from deepface import DeepFace

image_folder = 'hse_faces_miem'
embeddings_path = 'embeddings.npy'
index_path = 'image_index.index'
csv_path = "staff_photo.csv"

embeddings = np.load(embeddings_path)
index = faiss.read_index(index_path)
df = pd.read_csv(csv_path)

filename_to_name = dict(zip(df["filename"], df["name"]))


query_image_path = 'selfie.jpg'
query_embedding = DeepFace.represent(img_path=query_image_path, model_name="Facenet", enforce_detection=False)[0]['embedding']
query_embedding = np.array(query_embedding).reshape(1, -1)

k = 3
distances, indices = index.search(query_embedding, k)

for i, distance in zip(indices[0], distances[0]):
    filename = os.path.basename(os.listdir(image_folder)[i])
    name = filename_to_name.get(filename, "Неизвестный")
    print(f"ФИО: {name}, Файл: {filename}, Расстояние: {distance:.2f}")
