import faiss
import numpy as np


embeddings = np.load('embeddings.npy')

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

faiss.write_index(index, 'image_index.index')

print("Индекс успешно создан и сохранен.")

