from rank_bm25 import BM25Okapi
import numpy as np


class BM25Retriever:
    def __init__(self, documents):
        self.documents = documents
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)

    def search(self, query, k=5):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        top_k_indices = np.argsort(scores)[::-1][:k]
        return top_k_indices, scores
