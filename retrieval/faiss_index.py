import faiss
import numpy as np
import os


class FaissIndex:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)

    def add(self, embeddings):
        self.index.add(np.array(embeddings))

    def search(self, query_embedding, k=5):
        distances, indices = self.index.search(
            np.array([query_embedding]), k
        )
        return distances, indices

    def save(self, path="faiss_index.bin"):
        faiss.write_index(self.index, path)

    def load(self, path="faiss_index.bin"):
        if os.path.exists(path):
            self.index = faiss.read_index(path)
            return True
        return False
