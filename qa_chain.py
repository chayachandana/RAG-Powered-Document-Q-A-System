import os
from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda

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
    """Search each resume file separately by pre-filtering docs"""
    # Get ALL docs from index (large k to get everything)
    all_results = db.similarity_search(query, k=200)
    
    # Group by source file
    by_file = {}
    for doc in all_results:
        source = doc.metadata.get("source_file", "unknown")
        if source not in by_file:
            by_file[source] = []
        by_file[source].append(doc)
    
    print(f"Found files: {list(by_file.keys())}")
    
    # Take top k_per_file chunks from each file
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

# Test questions
questions = [
    "Who has machine learning experience?",
    "Who is the ML engineer?",
    "What is Chaya's background?",
    "Compare the two candidates",
    "Who has PyTorch experience?",
]

print("\n" + "="*50)
print("Testing RAG Q&A on multiple resumes")
print("="*50 + "\n")

for q in questions:
    print(f"Question: {q}")
    answer = chain.invoke(q)
    print(f"Answer: {answer}")
    print("-"*40)