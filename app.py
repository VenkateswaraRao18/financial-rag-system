import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ingestion.loader import load_pdfs
from ingestion.chunking import chunk_text
from embeddings.embedder import Embedder
from retrieval.faiss_index import FaissIndex
from retrieval.bm25_retriever import BM25Retriever
from retrieval.re_ranker import ReRanker
from generation.gemini_llm import generate_answer


def main():
    print("Loading documents...")
    docs = load_pdfs()

    all_chunks = []

    for doc in docs:
        chunks = chunk_text(doc["content"])
        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "source": doc["file_name"]
            })

    print(f"Total chunks: {len(all_chunks)}")

    embedder = Embedder()
    index = FaissIndex(384)

    if not index.load():
        print("Creating new FAISS index...")
        embeddings = embedder.embed([chunk["text"] for chunk in all_chunks])
        index.add(embeddings)
        index.save()
        print("FAISS index saved.")
    else:
        print("Loaded existing FAISS index.")

    bm25 = BM25Retriever([chunk["text"] for chunk in all_chunks])
    reranker = ReRanker()

    while True:
        question = input("\nAsk a question (or 'exit'): ")
        if question.lower() == "exit":
            break

        print("\n🔎 Performing Hybrid Retrieval...")

        # Dense Retrieval
        query_embedding = embedder.embed([question])[0]
        dense_distances, dense_indices = index.search(query_embedding, k=10)

        # BM25 Retrieval
        bm25_indices, bm25_scores = bm25.search(question, k=10)

        # Combine indices
        combined_indices = list(dense_indices[0]) + list(bm25_indices)

        seen = set()
        candidate_indices = []
        for idx in combined_indices:
            if idx not in seen:
                candidate_indices.append(idx)
                seen.add(idx)

        # Get candidate documents
        candidate_docs = [all_chunks[idx]["text"] for idx in candidate_indices]

        print("🔁 Re-ranking candidates...")

        # Re-rank
        top_docs = reranker.rerank(question, candidate_docs, top_k=5)

        print("\nTop Re-Ranked Chunks:\n")

        for doc in top_docs:
            print(doc[:400])
            print("-" * 80)

        context = "\n".join(top_docs)

        answer = generate_answer(context, question)

        print("\n💡 Answer:")
        print(answer)


if __name__ == "__main__":
    main()
