#%%
import os
import json
from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load FAISS index
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_index", embeddings,
                       allow_dangerous_deserialization=True)

# Set up Groq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

retriever = db.as_retriever(search_kwargs={"k": 4})

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

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ── Test cases with expected keywords ──
test_cases = [
    {
        "question": "Where did this person work?",
        "expected_keywords": ["tata", "george washington", "university"],
    },
    {
        "question": "What is this person's educational background?",
        "expected_keywords": ["master", "data science", "bachelor"],
    },
    {
        "question": "What deep learning models has this person used?",
        "expected_keywords": ["lstm", "gru", "transformer"],
    },
    {
        "question": "What cloud platforms has this person worked with?",
        "expected_keywords": ["aws", "gcp", "databricks"],
    },
    {
        "question": "What programming languages does this person know?",
        "expected_keywords": ["python", "sql"],
    },
    {
        "question": "What is this person's experience with anomaly detection?",
        "expected_keywords": ["anomaly", "isolation forest", "tata"],
    },
    {
        "question": "What frameworks does this person use?",
        "expected_keywords": ["pytorch", "tensorflow", "scikit"],
    },
    {
        "question": "What is this person's name?",
        "expected_keywords": ["chaya"],
    },
    {
        "question": "What deployment tools does this person know?",
        "expected_keywords": ["docker", "flask", "api"],
    },
    {
        "question": "What is RAG and has this person worked with it?",
        "expected_keywords": ["retrieval", "rag"],
    },
]

# ── Run evaluation ──
print("\n" + "="*60)
print("RAG EVALUATION REPORT")
print("="*60 + "\n")

results = []
passed = 0

for i, case in enumerate(test_cases):
    question = case["question"]
    keywords = case["expected_keywords"]

    answer = chain.invoke(question)
    answer_lower = answer.lower()

    # Check how many expected keywords appear in the answer
    matched = [kw for kw in keywords if kw.lower() in answer_lower]
    score = len(matched) / len(keywords)
    success = score >= 0.5  # pass if at least 50% keywords found

    if success:
        passed += 1

    results.append({
        "question": question,
        "answer": answer,
        "matched_keywords": matched,
        "score": round(score, 2),
        "passed": success
    })

    status = "✅ PASS" if success else "❌ FAIL"
    print(f"Q{i+1}: {question}")
    print(f"Answer: {answer[:150]}...")
    print(f"Keywords matched: {matched}")
    print(f"Score: {score:.0%} → {status}")
    print("-"*60)

# ── Summary ──
accuracy = passed / len(test_cases) * 100
print(f"\n{'='*60}")
print(f"FINAL RESULTS")
print(f"{'='*60}")
print(f"Total questions : {len(test_cases)}")
print(f"Passed          : {passed}")
print(f"Failed          : {len(test_cases) - passed}")
print(f"Accuracy        : {accuracy:.1f}%")

# Save results to JSON
with open("eval_results.json", "w") as f:
    json.dump({
        "accuracy": f"{accuracy:.1f}%",
        "passed": passed,
        "failed": len(test_cases) - passed,
        "results": results
    }, f, indent=2)

print(f"\nResults saved to eval_results.json")