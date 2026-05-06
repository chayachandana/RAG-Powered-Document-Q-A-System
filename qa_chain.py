#%%
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load the saved FAISS index
print("Loading FAISS index...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_index", embeddings,
                       allow_dangerous_deserialization=True)
print("Index loaded!")

# Set up free Groq LLM
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

# Helper to format retrieved docs
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build chain using modern LCEL syntax
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Test questions
questions = [
    "What ML skills does this person have?",
    "Where did this person work?",
    "What is this person's educational background?",
    "What deep learning models has this person used?"
]

print("\n" + "="*50)
print("Testing RAG Q&A on resume")
print("="*50 + "\n")

for q in questions:
    print(f"Question: {q}")
    answer = chain.invoke(q)
    print(f"Answer: {answer}")
    print("-"*40)
       
# %%
