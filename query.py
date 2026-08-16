from retrieval import query_collection, bm25_search, build_bm25_index, get_all_chunks_from_db, merge_rrf
from generation import build_prompt
from embedding_manager import EmbeddingManager
from google import genai
import os
from dotenv import load_dotenv

# MUST be loaded BEFORE instantiating genai.Client()
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

query_text = "what is Para Virtualization?"
n_results = 3
embedder = EmbeddingManager()
client = genai.Client()

all_chunks = get_all_chunks_from_db()
bm25_index = build_bm25_index(all_chunks)

# keyword search
bm25_results = bm25_search(query_text, bm25_index, all_chunks, n_results)
# print("BM25 RESULTS:")
# for i, (chunk, score) in enumerate(bm25_results):
#     print(f"{i}: score={score:.2f} | {chunk[:100]}")

# vector search
vector_results  = query_collection(query_text, embedder, n_results)['documents'][0]
# print("Vector Results: ")
# for i, chunk in enumerate(vector_results):
#     print(f"{i}: {chunk[:100]}")  # first 100 chars, just to preview

final_results = merge_rrf(vector_results, bm25_results)
# print(f"this are the final results: ", final_results)

prompt = build_prompt(query_text, final_results)

# LLM Call
response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents=prompt
)
print(f" this is the {response.text}")