# import os
# import time

# from ingestion.loader import load_pdfs
# from ingestion.chunking import chunk_text
# from embeddings.embedder import Embedder
# from retrieval.faiss_index import FaissIndex
# from retrieval.bm25_retriever import BM25Retriever
# from retrieval.re_ranker import ReRanker
# from generation.gemini_llm import generate_answer


# class RAGPipeline:

#     def __init__(self):
#         self.embedder = Embedder()
#         self.index = FaissIndex(384)
#         self.reranker = ReRanker()

#         self.all_chunks = []
#         self.bm25 = None

#     def build_index(self):
#         docs = load_pdfs()

#         for doc in docs:
#             chunks = chunk_text(doc["content"])
#             for chunk in chunks:
#                 self.all_chunks.append({
#                     "text": chunk,
#                     "source": doc["file_name"]
#                 })

#         if not self.index.load():
#             embeddings = self.embedder.embed(
#                 [chunk["text"] for chunk in self.all_chunks]
#             )
#             self.index.add(embeddings)
#             self.index.save()

#         self.bm25 = BM25Retriever(
#             [chunk["text"] for chunk in self.all_chunks]
#         )

#     def ask(self, question):

#         query_embedding = self.embedder.embed([question])[0]
#         dense_distances, dense_indices = self.index.search(query_embedding, k=10)

#         bm25_indices, _ = self.bm25.search(question, k=10)

#         combined_indices = list(dense_indices[0]) + list(bm25_indices)

#         seen = set()
#         candidate_indices = []
#         for idx in combined_indices:
#             if idx not in seen:
#                 candidate_indices.append(idx)
#                 seen.add(idx)

#         candidate_docs = [
#             self.all_chunks[idx]["text"] for idx in candidate_indices
#         ]

#         top_docs = self.reranker.rerank(
#             question, candidate_docs, top_k=5
#         )

#         context = "\n".join(top_docs)

#         answer = generate_answer(context, question)

#         return answer




import os
import shutil

from ingestion.loader import load_pdfs
from ingestion.chunking import chunk_text
from embeddings.embedder import Embedder
from retrieval.faiss_index import FaissIndex
from retrieval.bm25_retriever import BM25Retriever
from retrieval.re_ranker import ReRanker
from generation.gemini_llm import generate_answer


class RAGPipeline:

    def __init__(self):
        self.embedder = Embedder()
        self.index = FaissIndex(384)
        self.reranker = ReRanker()

        self.all_chunks = []
        self.bm25 = None

    def build_index(self, data_path="uploaded_docs"):

        # Reset everything
        self.all_chunks = []

        # Delete old FAISS index file if exists
        if os.path.exists("faiss_index.bin"):
            os.remove("faiss_index.bin")

        docs = load_pdfs(data_path)

        for doc in docs:
            chunks = chunk_text(doc["content"])
            for chunk in chunks:
                self.all_chunks.append({
                    "text": chunk,
                    "source": doc["file_name"]
                })

        if not self.all_chunks:
            return

        embeddings = self.embedder.embed(
            [chunk["text"] for chunk in self.all_chunks]
        )

        self.index = FaissIndex(384)
        self.index.add(embeddings)
        self.index.save()

        self.bm25 = BM25Retriever(
            [chunk["text"] for chunk in self.all_chunks]
        )

    def ask(self, question):

        if not self.all_chunks:
            return "No document uploaded yet."

        query_embedding = self.embedder.embed([question])[0]
        dense_distances, dense_indices = self.index.search(query_embedding, k=10)

        bm25_indices, _ = self.bm25.search(question, k=10)

        combined_indices = list(dense_indices[0]) + list(bm25_indices)

        seen = set()
        candidate_indices = []
        for idx in combined_indices:
            if idx not in seen:
                candidate_indices.append(idx)
                seen.add(idx)

        candidate_docs = [
            self.all_chunks[idx]["text"] for idx in candidate_indices
        ]

        top_docs = self.reranker.rerank(
            question, candidate_docs, top_k=5
        )

        context = "\n".join(top_docs)

        answer = generate_answer(context, question)

        return answer
