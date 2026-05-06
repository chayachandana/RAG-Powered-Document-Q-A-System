#%%
import os
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

app = Flask(__name__)

# Load FAISS index
print("Loading FAISS index...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_index", embeddings,
                       allow_dangerous_deserialization=True)

#%%
# Set up Groq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# Retriever
retriever = db.as_retriever(search_kwargs={"k": 4})

# Prompt
prompt = PromptTemplate.from_template("""
Use the following context to answer the question.
If the answer is not in the context, say "I don't know based on the provided document."

Context:
{context}

Question:
{question}

Answer:
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build chain
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("App ready!")

# ── Routes ──

@app.route("/")
def home():
    return jsonify({
        "message": "RAG Q&A API is running",
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

if __name__ == "__main__":
    app.run(debug=True, port=8080)