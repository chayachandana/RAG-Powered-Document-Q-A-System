# 💼 RAG-Powered Resume Q&A System

A production-ready document intelligence system that lets you query multiple resumes in natural language. Ask questions like *"Who has PyTorch experience?"* or *"Compare the two candidates"* and get accurate, grounded answers pulled directly from the source documents.

Built to solve a real retrieval problem: standard FAISS similarity search systematically favors longer documents over shorter ones in multi-document corpora. This system fixes that with a per-file chunk grouping strategy that ensures every resume gets equal representation in the context window.

**Live API:** [https://rag-powered-document-q-a-system-production.up.railway.app]

---

## 🎥 Demo



---

## 🧠 How It Works

```
INGEST (once)
PDF files → PyPDFLoader → RecursiveCharacterTextSplitter (300 chars, 100 overlap)
        → HuggingFace Embeddings (all-MiniLM-L6-v2) → FAISS Index (saved to disk)

QUERY (every question)
User question → Embed question → FAISS search (top 200)
             → Group by source file → Take top 4 per file
             → Build prompt with context → Groq LLaMA 3.3 70B → Answer
```

The key engineering decision: instead of taking the global top-k results (which would be dominated by the longest document), the system fetches results grouped by source file and takes the best chunks from each — ensuring the LLM always sees context from every resume before answering.

---

## 📁 Folder Structure

```
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
```

---

## ⚙️ Tech Stack

| Component | Tool |
|---|---|
| PDF Loading | LangChain + PyPDF |
| Text Splitting | RecursiveCharacterTextSplitter |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (free, local) |
| Vector Store | FAISS |
| LLM | Groq `llama-3.3-70b-versatile` (free) |
| API | Flask |
| Evaluation | Custom keyword-match scoring |

---

## 🚀 Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/chayachandana/RAG-Powered-Document-Q-A-System.git
cd RAG-Powered-Document-Q-A-System
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up API keys

Create a `.env` file in the root folder:
```
GROQ_API_KEY=your_groq_key_here
```

Get a free Groq API key at [console.groq.com](https://console.groq.com) — no credit card needed.

### 5. Add your PDFs

Drop any PDF files into the `resumes/` folder.

### 6. Build the index
```bash
python ingest.py
```

Expected output:
```
Loading candidate_1.pdf...
  ✓ Loaded 1 pages from candidate_1.pdf
Loading candidate_2.pdf...
  ✓ Loaded 2 pages from candidate_2.pdf

Total: 3 pages loaded from resumes
Split into 52 chunks
Embedding model loaded!
Done! faiss_index folder created successfully.
```

### 7. Run the app
```bash
python app.py
```

Open your browser at: **http://localhost:8080/chat**

---

## 💬 Example Questions

| Question | What it tests |
|---|---|
| `Who has PyTorch experience?` | Skill lookup across resumes |
| `Compare the two candidates` | Multi-document reasoning |
| `Who worked at a top tech company?` | Experience comparison |
| `What is Chaya's educational background?` | Single-candidate lookup |
| `Who is stronger in machine learning?` | Analytical comparison |

---

## 📊 Evaluation Results

Ran structured evaluation across 10 domain-specific questions with keyword-match scoring:

```
Total questions : 10
Passed          : 9
Failed          : 1
Accuracy        : 90.0%
```

Full results saved in `eval_results.json`. The one failure (education question) was traced to a chunking boundary issue — the Master's degree mention was split across two chunks. Fixed by reducing chunk size from 500→300 and increasing overlap from 50→100.

To re-run evaluation:
```bash
python eval.py
```

---

## 🐛 Known Issues & How I Fixed Them

**Problem 1 — Retrieval bias across documents**
FAISS similarity search was returning chunks almost entirely from Chandan's resume (longer document) even when asking about Chaya. Standard top-k retrieval doesn't guarantee representation from all sources.

**Fix:** Implemented per-file chunk grouping — fetch top 200 results globally, group by `source_file` metadata, then take top-k from each group before building the context window.

**Problem 2 — Chunking boundary failures**
Education section answers were failing because degree information was split across chunk boundaries.

**Fix:** Reduced chunk size from 500→300 chars and increased overlap from 50→100 chars to preserve semantic continuity across boundaries.

---

## ☁️ Deployed on Railway

The backend API is live. Test it instantly without running anything locally.

### Test with curl
```bash
curl -X POST https://rag-powered-document-q-a-system-production.up.railway.app/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Who has machine learning experience?"}'
```

### Test with Postman
1. Download **Postman** at [postman.com](https://postman.com) — free
2. Click **New Request**
3. Change **GET → POST**
4. URL: `https://rag-powered-document-q-a-system-production.up.railway.app/ask`
5. Click **Body → raw → JSON**
6. Paste:
```json
{
  "question": "Who has machine learning experience?"
}
```
7. Click **Send**

### Deploy your own instance
1. Fork this repo
2. Go to [railway.app](https://railway.app) → 
3. Connect your forked repo
4. Add environment variable: `GROQ_API_KEY` = your key
5. Commit your `faiss_index/` folder so it ships with the code:
```bash
git add faiss_index/
git commit -m "add faiss index for deployment"
git push
```
1. Make sure the last line of `app.py` reads:
```python
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
```
Railway auto-detects Python and deploys automatically.

## 🔑 Environment Variables

| Variable | Required | Where to get it |
|---|---|---|
| `GROQ_API_KEY` | Yes | [console.groq.com](https://console.groq.com) |

---

## 📄 License

MIT — use freely, attribution appreciated.

---

## 🙋 Author

**Chaya Chandana Doddaiggaluru Appajigowda**
[LinkedIn](your-linkedin-url) · [GitHub](https://github.com/chayachandana)
