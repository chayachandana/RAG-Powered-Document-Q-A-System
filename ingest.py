#%%
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

pdf_folder = "data"
all_docs = []

for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        print(f"Loading {filename}...")
        loader = PyPDFLoader(os.path.join(pdf_folder, filename))
        docs = loader.load()
        # Tag each chunk with which resume it came from
        for doc in docs:
            doc.metadata["source_file"] = filename
        all_docs.extend(docs)

print(f"Loaded {len(all_docs)} pages total from {pdf_folder}")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
chunks = splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks")

# Create embeddings using HuggingFace (free, runs locally)
print("Loading embedding model... first time takes ~1 min to download")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Embedding model loaded!")

# Create and save FAISS index
print("Creating FAISS index...")
db = FAISS.from_documents(chunks, embeddings)
db.save_local("faiss_index")

print("Done! faiss_index folder created successfully.")
# %%
