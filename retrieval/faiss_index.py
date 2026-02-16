import faiss
import numpy as np


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
