import os
from deepface import DeepFace
import numpy as np

# Папка с изображениями
image_folder = 'hse_faces_miem'

# Список путей к изображениям
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpeg') or f.endswith('.jpg')]

# Список эмбеддингов
embeddings = []

# Извлечение эмбеддингов для всех изображений
for image_path in image_paths:
    embedding = DeepFace.represent(img_path=image_path, model_name="Facenet", enforce_detection=False)
    embeddings.append(embedding[0]['embedding'])

# Преобразуем эмбеддинги в массив numpy
embeddings = np.array(embeddings)

# Сохраним эмбеддинги для использования в дальнейшем
np.save('embeddings.npy', embeddings)

print("Эмбеддинги успешно извлечены и сохранены.")
