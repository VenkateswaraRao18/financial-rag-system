# 📊 Financial Report RAG System

An enterprise-grade Retrieval-Augmented Generation (RAG) system built to analyze financial reports (10-K filings) and answer analytical business questions using hybrid retrieval and large language models.

This project simulates an internal AI Financial Analyst used in enterprise environments.

---

## 🚀 Project Objective

Build a production-style RAG pipeline capable of:

- Processing large financial documents (10-K reports)
- Performing vector-based semantic retrieval
- Generating grounded answers using Gemini LLM
- Providing explainable and modular architecture
- Scaling toward hybrid retrieval and API deployment

---

## 🏗 System Architecture

Financial PDFs (10-K Reports)
→
PDF Loader (PyPDF)
→
Sliding Window Chunking
→
SentenceTransformers Embeddings (MiniLM)
→
FAISS Vector Index (Dense Retrieval)
→
Top-K Relevant Chunks
→
Gemini LLM (Context-Grounded Generation)
→
Final Answer

---

## 🔍 Core Components

### 1️⃣ Document Ingestion

- Extracts text from financial PDFs
- Handles multi-page annual reports

### 2️⃣ Chunking Strategy

- Sliding window chunking
- Overlap-based context preservation
- Optimized for long financial paragraphs

### 3️⃣ Embeddings

- Model: `all-MiniLM-L6-v2`
- Lightweight and CPU-efficient
- Generates dense vector representations

### 4️⃣ Vector Search

- FAISS (L2 similarity)
- Top-K semantic retrieval
- Efficient search over 900+ chunks

### 5️⃣ LLM Generation

- Model: Gemini (Cloud-based)
- Prompt constrained to retrieved context
- Prevents hallucination outside report

---

## 🧠 Example Analytical Questions

- What were the main revenue drivers in 2024?
- What key risk factors were identified?
- Compare automotive revenue with energy segment performance.
- How does the company describe liquidity and debt obligations?

---

## 📦 Tech Stack

- Python
- SentenceTransformers
- FAISS (CPU)
- Google Gemini API
- Modular Project Architecture
- VS Code Development Environment

---

## 🏆 Engineering Highlights

- Modular folder structure
- Separation of ingestion, embeddings, retrieval, and generation
- Debug-friendly retrieval pipeline
- Source tracking for retrieved chunks
- Context-grounded prompting
- CPU-friendly architecture

---

## 📈 Current Performance

- ~900+ chunks indexed
- Sub-second retrieval time
- Accurate financial answer grounding
- No external hallucination observed during testing

---

## 🔥 Upcoming Enhancements

### 🔹 Hybrid Retrieval (BM25 + Dense)

Combine lexical + semantic search for improved recall.

### 🔹 Cross-Encoder Re-ranking

Transformer-based re-ranking of retrieved chunks.

### 🔹 FAISS Persistence

Save and reload index to avoid recomputation.

### 🔹 API Deployment

Expose system via FastAPI endpoint.

### 🔹 Evaluation Framework

Integrate RAGAS for:

- Faithfulness
- Answer relevancy
- Context precision

### 🔹 Monitoring

Add logging for:

- Retrieval latency
- Token usage
- Drift detection

---

## 🌍 Production Roadmap

This project is designed to evolve toward:

- Enterprise-scale knowledge assistant
- Internal AI financial research tool
- Scalable cloud deployment (AWS / GCP)
- Multi-document hybrid retrieval

---

## 📂 Project Structure

financial-rag/

├── data/

├── ingestion/

├── embeddings/

├── retrieval/

├── generation/

├── app.py

└── README.md

---

## 🎯 Why This Project Matters

This implementation demonstrates:

- Applied NLP engineering
- System design thinking
- Production-level code organization
- Real-world business use case
- LLM integration in enterprise context

---

## 👨‍💻 Author

Built as part of advanced applied NLP and system design learning focused on product-based company standards.
