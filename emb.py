import os
import numpy as np
from deepface import DeepFace

image_folder = 'hse_faces_miem'
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpeg', '.jpg', '.png'))]

embeddings = []
for image_path in image_paths:
    embedding = DeepFace.represent(img_path=image_path, model_name="Facenet", enforce_detection=False)[0]['embedding']
    embeddings.append(embedding)

embeddings = np.array(embeddings)
np.save('embeddings.npy', embeddings)

print("Эмбеддинги успешно извлечены и сохранены.")
