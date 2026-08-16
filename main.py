import os
from dotenv import load_dotenv

# 1. ALWAYS LOAD THE ENV FIRST
load_dotenv(override=True)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 2. Check that key exists before running processing logic
if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY not found in .env file!")

from chunking import read_documents, split_documents, FOLDER_PATH, CHUNK_SIZE, CHUNK_OVERLAP
from embedding_manager import EmbeddingManager
from google import genai
from chroma_db import add_chunks_to_collection

client = genai.Client()

# MAIN PIPELINE

if __name__ == "__main__": # this code only runs when the file is executed directly
    # Step 1: load raw documents from folder
    documents = read_documents(FOLDER_PATH)

    # Step 2: split documents into token-based chunks
    chunks = split_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"Total chunks created: {len(chunks)}")

    # Step 3: embed all chunks
    embedder = EmbeddingManager()
    chunk_texts = [c.page_content for c in chunks]   # extract raw text from each chunk
    embeddings = embedder.embed_chunks_batch(chunk_texts)
    print(f"Total embeddings created: {len(embeddings)}")
    # embedding are generated on the basis of only main page content metadata is not included here

    add_chunks_to_collection(chunks, embeddings)

    # Step 4: preview first chunk + its embedding
    print("\n--- Preview ---")
    print("Chunk text:", chunk_texts[0][:200])   # first 200 chars
    print("Embedding (first 5 dims):", embeddings[0][:5])