import faiss
import numpy as np

# Загружаем эмбеддинги
embeddings = np.load('embeddings.npy')

# Создадим индекс
dimension = embeddings.shape[1]  # Размерность эмбеддингов
index = faiss.IndexFlatL2(dimension)

# Добавим эмбеддинги в индекс
index.add(embeddings)

# Сохраним индекс для использования
faiss.write_index(index, 'image_index.index')

print("Индекс успешно создан и сохранен.")

