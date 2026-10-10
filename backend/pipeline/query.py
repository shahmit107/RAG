# Actual Application of the RAG pipeline: it is executed by the sever.py
from google import genai
from dotenv import load_dotenv
from backend.paths import ENV_PATH
import os
from backend.pipeline.generation import build_prompt
from backend.pipeline.embedding_manager import EmbeddingManager
from sentence_transformers import CrossEncoder
from backend.pipeline.retrieval import *
from groq import Groq

load_dotenv(ENV_PATH)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

print("Execution Started")

class RAGPipeline:
    def __init__(self):
        # Load these ONCE when server starts
        self.embedder = EmbeddingManager()

        # Load once, not per-query — this loads the model into memory
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        self.all_chunks = get_all_chunks_from_db()  # as we are performing the hybrid retrieval

        self.bm25_index = build_bm25_index(self.all_chunks)

        self.client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"timeout": 30000, "retry_options": {"attempts": 1}},
        )
        self.groq_client = Groq(api_key=GROQ_API_KEY, timeout=30.0, max_retries=0)

    def run(self, query_text):

        vector_search = query_collection(query_text, self.embedder, 3) # gives the format of the chroma db
        vector_chunks = [
            {"document": doc, "metadata": meta}
            for doc, meta in zip(vector_search["documents"][0], vector_search["metadatas"][0])
        ]

        bm25_results_keyword_search = bm25_search(query_text, self.bm25_index, self.all_chunks, 10)

        final_results = merge_rrf(vector_chunks, bm25_results_keyword_search)

        reranked_results = rerank(query_text, final_results, self.cross_encoder, top_n=4)

        print(f"This are the {reranked_results}")

        prompt = build_prompt(query_text, reranked_results)

        fallbacks = [
            ("gemini", "gemini-3.5-flash-lite"),
            ("groq", "openai/gpt-oss-20b"),
        ]

        for provider, model_name in fallbacks:
            try:
                if provider == "gemini":
                    response = self.client.models.generate_content(model=model_name, contents=prompt)
                    if response.text:
                        return response.text
                elif provider == "groq":
                    response = self.groq_client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    if response.choices[0].message.content:
                        return response.choices[0].message.content
            except Exception as e:
                print(f"{provider}/{model_name} failed: {e}")
                continue

        return "All models are currently unavailable. Please try again later."

if __name__ == "__main__":
    rag = RAGPipeline()
    print("before fun call")
    rag.run("what is the para virtualisation?")
    print("after fun call")
