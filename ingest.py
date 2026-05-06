#%%
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load all PDFs from the "data" folder
pdf_folder = "resumes"
all_docs = []

for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        try:
            print(f"Loading {filename}...")
            loader = PyPDFLoader(os.path.join(pdf_folder, filename))
            docs = loader.load()

            if not docs:
                print(f"  ⚠ Skipping {filename} — no content found")
                continue

            for doc in docs:
                doc.metadata["source_file"] = filename
            all_docs.extend(docs)
            print(f"  ✓ Loaded {len(docs)} pages")

        except Exception as e:
            print(f"  ✗ Skipping {filename} — error: {e}")

print(f"\nTotal: {len(all_docs)} pages loaded from {pdf_folder}")

if not all_docs:
    print("No documents loaded — check your data folder!")
    exit()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
chunks = splitter.split_documents(all_docs)
print(f"Split into {len(chunks)} chunks")

# Create embeddings
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Embedding model loaded!")

# Create and save FAISS index
print("Creating FAISS index...")
db = FAISS.from_documents(chunks, embeddings)
db.save_local("faiss_index")

print("Done! faiss_index folder created successfully.")
# %%