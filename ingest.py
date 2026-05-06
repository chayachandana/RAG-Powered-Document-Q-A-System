#%%
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load your PDF
loader = PyPDFLoader("data/Chaya_Chandana_Doddaiggaluru_Appajigowda.pdf")
docs = loader.load()
print(f"Loaded {len(docs)} pages from PDF")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
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
