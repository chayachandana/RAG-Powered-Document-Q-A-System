import os
from dotenv import load_dotenv
load_dotenv()

import webbrowser
from flask import Flask, request, jsonify
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda

app = Flask(__name__)

# Load FAISS index
print("Loading FAISS index...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_index", embeddings,
                       allow_dangerous_deserialization=True)
print("Index loaded!")

# Set up Groq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

def get_all_chunks(query, k_per_file=4):
    """Search all docs then group by source file"""
    all_results = db.similarity_search(query, k=200)

    # Group by source file
    by_file = {}
    for doc in all_results:
        source = doc.metadata.get("source_file", "unknown")
        if source not in by_file:
            by_file[source] = []
        by_file[source].append(doc)

    print(f"Found files: {list(by_file.keys())}")

    # Take top k_per_file from each file
    combined = []
    for source, docs in by_file.items():
        combined.extend(docs[:k_per_file])

    return combined

def format_docs(docs):
    result = []
    for doc in docs:
        source = doc.metadata.get("source_file", "unknown")
        result.append(f"[Resume: {source}]\n{doc.page_content}")
    return "\n\n".join(result)

def rag_chain_fn(question):
    chunks = get_all_chunks(question, k_per_file=4)
    context = format_docs(chunks)
    prompt_text = f"""
You are a resume screening assistant with access to multiple candidate resumes.
Always answer using information from ALL resumes in the context, not just one.
Each chunk is tagged with [Resume: filename] so you know which candidate it belongs to.
Figure out the candidate's real name from the resume content itself.
If asked about a specific person use their actual name not the filename.
If comparing candidates mention each by their real name.
If the answer is not in the context say "I don't know based on the provided resumes."

Context:
{context}

Question:
{question}

Answer:"""
    response = llm.invoke(prompt_text)
    return response.content

chain = RunnableLambda(rag_chain_fn)

print("App ready!")

# ── Routes ──

@app.route("/")
def home():
    return jsonify({
        "message": "Resume Q&A API is running",
        "usage": "POST /ask with JSON body: {question: your question here}"
    })

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()

    if not data or "question" not in data:
        return jsonify({"error": "Please provide a question in JSON body"}), 400

    question = data["question"]

    if not question.strip():
        return jsonify({"error": "Question cannot be empty"}), 400

    try:
        answer = chain.invoke(question)
        return jsonify({
            "question": question,
            "answer": answer,
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/chat")
def chat_ui():
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Resume Q&A Chatbot</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #0f0f0f; color: #fff; height: 100vh;
            display: flex; flex-direction: column;
            align-items: center; justify-content: center;
        }
        .container {
            width: 100%; max-width: 780px; height: 100vh;
            display: flex; flex-direction: column; padding: 20px;
        }
        .header { text-align: center; padding: 24px 0 16px; }
        .header h1 { font-size: 22px; font-weight: 600; color: #fff; letter-spacing: -0.3px; }
        .header p { font-size: 13px; color: #666; margin-top: 6px; }
        .chat-box {
            flex: 1; overflow-y: auto; padding: 20px 0;
            display: flex; flex-direction: column; gap: 20px;
            scrollbar-width: thin; scrollbar-color: #333 transparent;
        }
        .message { display: flex; gap: 12px; align-items: flex-start; animation: fadeIn 0.3s ease; }
        @keyframes fadeIn { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
        .avatar {
            width: 32px; height: 32px; border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-size: 14px; flex-shrink: 0; margin-top: 2px;
        }
        .avatar.user { background: #2563eb; }
        .avatar.bot  { background: #18181b; border: 1px solid #333; }
        .bubble {
            max-width: 85%; padding: 12px 16px; border-radius: 16px;
            font-size: 14px; line-height: 1.7;
        }
        .user .bubble { background: #1e3a5f; color: #e2eeff; border-bottom-right-radius: 4px; }
        .bot  .bubble { background: #18181b; color: #e4e4e7; border: 1px solid #27272a; border-bottom-left-radius: 4px; }
        .typing {
            display: flex; gap: 5px; padding: 14px 16px;
            background: #18181b; border: 1px solid #27272a;
            border-radius: 16px; border-bottom-left-radius: 4px; width: fit-content;
        }
        .dot { width: 7px; height: 7px; background: #555; border-radius: 50%; animation: bounce 1.2s infinite; }
        .dot:nth-child(2) { animation-delay: .2s; }
        .dot:nth-child(3) { animation-delay: .4s; }
        @keyframes bounce {
            0%,60%,100% { transform: translateY(0); }
            30% { transform: translateY(-6px); background: #888; }
        }
        .suggestions { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 4px; }
        .suggestion-btn {
            background: #18181b; border: 1px solid #333; color: #a1a1aa;
            padding: 7px 14px; border-radius: 20px; font-size: 12px;
            cursor: pointer; transition: all 0.2s;
        }
        .suggestion-btn:hover { background: #27272a; color: #fff; border-color: #555; }
        .input-area { padding: 16px 0 8px; border-top: 1px solid #1e1e1e; }
        .input-row {
            display: flex; gap: 10px; align-items: flex-end;
            background: #18181b; border: 1px solid #27272a;
            border-radius: 16px; padding: 10px 14px; transition: border-color 0.2s;
        }
        .input-row:focus-within { border-color: #2563eb; }
        textarea {
            flex: 1; background: transparent; border: none; outline: none;
            color: #fff; font-size: 14px; resize: none;
            max-height: 120px; min-height: 24px; line-height: 1.5; font-family: inherit;
        }
        textarea::placeholder { color: #444; }
        .send-btn {
            background: #2563eb; border: none; color: #fff;
            width: 34px; height: 34px; border-radius: 10px; cursor: pointer;
            display: flex; align-items: center; justify-content: center;
            transition: background 0.2s; flex-shrink: 0;
        }
        .send-btn:hover { background: #1d4ed8; }
        .send-btn:disabled { background: #27272a; cursor: not-allowed; }
        .footer { text-align: center; font-size: 11px; color: #333; padding: 8px 0; }
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>💼 Resume Q&A</h1>
        <p>Ask anything about the candidates — or ask me to compare them!</p>
    </div>

    <div class="chat-box" id="chatBox">
        <div class="message bot">
            <div class="avatar bot">🤖</div>
            <div>
                <div class="bubble">
                    Hi! I have access to multiple candidate resumes. Ask me about anyone\'s skills, experience, or education — or ask me to compare candidates!
                </div>
                <div class="suggestions" id="suggestions">
                    <button class="suggestion-btn" onclick="askSuggestion(this)">Who has ML experience?</button>
                    <button class="suggestion-btn" onclick="askSuggestion(this)">Compare the two candidates</button>
                    <button class="suggestion-btn" onclick="askSuggestion(this)">Who has PyTorch experience?</button>
                    <button class="suggestion-btn" onclick="askSuggestion(this)">Who worked at a top company?</button>
                </div>
            </div>
        </div>
    </div>

    <div class="input-area">
        <div class="input-row">
            <textarea id="userInput" placeholder="Ask about any candidate..."
                rows="1" onkeydown="handleKey(event)" oninput="autoResize(this)"></textarea>
            <button class="send-btn" id="sendBtn" onclick="sendMessage()">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                    <line x1="22" y1="2" x2="11" y2="13"></line>
                    <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                </svg>
            </button>
        </div>
    </div>
    <div class="footer">Powered by RAG · LangChain · Groq · FAISS</div>
</div>

<script>
    function autoResize(el) {
        el.style.height = "auto";
        el.style.height = Math.min(el.scrollHeight, 120) + "px";
    }
    function handleKey(e) {
        if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
    }
    function askSuggestion(btn) {
        document.getElementById("userInput").value = btn.textContent;
        document.getElementById("suggestions").style.display = "none";
        sendMessage();
    }
    function addMessage(text, role) {
        const box = document.getElementById("chatBox");
        const div = document.createElement("div");
        div.className = "message " + role;
        div.innerHTML =
            "<div class=\'avatar " + role + "\'>" + (role === "user" ? "👤" : "🤖") + "</div>" +
            "<div class=\'bubble\'>" + text.replace(/\\n/g, "<br>") + "</div>";
        box.appendChild(div);
        box.scrollTop = box.scrollHeight;
    }
    function showTyping() {
        const box = document.getElementById("chatBox");
        const div = document.createElement("div");
        div.className = "message bot";
        div.id = "typing";
        div.innerHTML =
            "<div class=\'avatar bot\'>🤖</div>" +
            "<div class=\'typing\'><div class=\'dot\'></div><div class=\'dot\'></div><div class=\'dot\'></div></div>";
        box.appendChild(div);
        box.scrollTop = box.scrollHeight;
    }
    function removeTyping() {
        const t = document.getElementById("typing");
        if (t) t.remove();
    }
    async function sendMessage() {
        const input = document.getElementById("userInput");
        const btn = document.getElementById("sendBtn");
        const question = input.value.trim();
        if (!question) return;
        addMessage(question, "user");
        input.value = "";
        input.style.height = "auto";
        btn.disabled = true;
        showTyping();
        try {
            const res = await fetch("/ask", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question })
            });
            const data = await res.json();
            removeTyping();
            addMessage(data.answer || data.error, "bot");
        } catch (err) {
            removeTyping();
            addMessage("Something went wrong. Please try again.", "bot");
        }
        btn.disabled = false;
        input.focus();
    }
</script>
</body>
</html>
    '''

if __name__ == "__main__":
    webbrowser.open("http://localhost:8080/chat")
    app.run(debug=True, port=8080)