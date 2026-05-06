# RAG-Powered Resume Q&A System

A production-ready document intelligence system that lets you query multiple resumes in natural language. Ask questions like "Who has PyTorch experience?" or "Compare the two candidates" and get accurate, grounded answers pulled directly from the source documents.
Built to solve a real retrieval problem: standard FAISS similarity search systematically favors longer documents over shorter ones in multi-document corpora. This system fixes that with a per-file chunk grouping strategy that ensures every resume gets equal representation in the context window.

# Live Demo: 

# 🧠 How It Works

INGEST (once)
PDF files → PyPDFLoader → RecursiveCharacterTextSplitter (300 chars, 100 overlap)
        → HuggingFace Embeddings (all-MiniLM-L6-v2) → FAISS Index (saved to disk)

QUERY (every question)
User question → Embed question → FAISS search (top 200)
             → Group by source file → Take top 4 per file
             → Build prompt with context → Groq LLaMA 3.3 70B → Answer

The key engineering decision: instead of taking the global top-k results (which would be dominated by the longest document), the system fetches results grouped by source file and takes the best chunks from each ensuring the LLM always sees context from every resume before answering.

## 📁 Folder Structure

RAG-Powered-Document-Q-A-System/
│
├── resumes/                         ← drop your PDF files here
│   ├── candidate_1.pdf
│   └── candidate_2.pdf
│
├── faiss_index/                     ← auto-generated, do not edit
│   ├── index.faiss
│   └── index.pkl
│
├── ingest.py                        ← step 1: load PDFs → build FAISS index
├── qa_chain.py                      ← step 2: test Q&A in terminal
├── app.py                           ← step 3: run Flask API + chat UI
├── eval.py                          ← evaluate retrieval accuracy
├── eval_results.json                ← saved evaluation results (90% accuracy)
├── test.py                          ← debug index contents
│
├── requirements.txt                 ← all dependencies
├── .env                             ← your API keys (never commit this)
├── .gitignore
└── README.md

## ⚙️ Tech Stack

 