import os
from google import genai
from dotenv import load_dotenv

# CONFIG
EMBEDDING_MODEL = "gemini-embedding-001" # Google's current embedding model
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# EMBEDDING MANAGER CLASS
class EmbeddingManager:
    def __init__(self, model=EMBEDDING_MODEL):
        self.model = model
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def embed_chunk(self, text):
        """Embeds a single piece of text, returns one vector."""
        response = self.client.models.embed_content(
            model=self.model,
            contents=text
        )
        return response.embeddings[0].values

    def embed_chunks_batch(self, chunk_list):
        """Embeds multiple chunks in a single API call."""
        response = self.client.models.embed_content(
            model=self.model,
            contents=chunk_list
        )
        return [embedding.values for embedding in response.embeddings]