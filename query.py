from retrieval import query_collection
from generation import build_prompt
from embedding_manager import EmbeddingManager
from google import genai
import os
from dotenv import load_dotenv

# MUST be loaded BEFORE instantiating genai.Client()
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

query_text = "what is full Virtualization?"
n_results = 3
embedder = EmbeddingManager()
client = genai.Client()

retrieved_chunks = query_collection(query_text, embedder, n_results)
prompt = build_prompt(query_text, retrieved_chunks['documents'][0])

# LLM Call
response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents=prompt
)
print(f" this is the {response.text}")