import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from ingestion.loader import load_pdfs
from ingestion.chunking import chunk_text
from embeddings.embedder import Embedder
from retrieval.faiss_index import FaissIndex
from generation.gemini_llm import generate_answer


def main():
    print("Loading documents...")
    docs = load_pdfs()

    all_chunks = []
    for doc in docs:
        chunks = chunk_text(doc["content"])
        all_chunks.extend(chunks)

    print(f"Total chunks: {len(all_chunks)}")

    embedder = Embedder()
    embeddings = embedder.embed(all_chunks)

    index = FaissIndex(len(embeddings[0]))
    index.add(embeddings)

    while True:
        question = input("\nAsk a question (or 'exit'): ")
        if question == "exit":
            break

        query_embedding = embedder.embed([question])[0]
        distances, indices = index.search(query_embedding)

        retrieved_chunks = [all_chunks[i] for i in indices[0]]

        context = "\n".join(retrieved_chunks)

        answer = generate_answer(context, question)

        print("\nAnswer:")
        print(answer)


if __name__ == "__main__":
    main()
