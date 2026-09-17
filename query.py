# Actual Application of the RAG pipeline: it is executed by the sever.py
from google import genai
from dotenv import load_dotenv
import os
from generation import build_prompt
from embedding_manager import EmbeddingManager
from sentence_transformers import CrossEncoder
from retrieval import *

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

print("Execution Started")

class RAGPipeline:
    def __init__(self):
        # Load these ONCE when server starts
        self.embedder = EmbeddingManager()

        # Load once, not per-query — this loads the model into memory
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        self.all_chunks = get_all_chunks_from_db()  # as we are performing the hybrid retrieval

        self.bm25_index = build_bm25_index(self.all_chunks)

        self.client = genai.Client()

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

        response = self.client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        return response.text # send the actual required resp.

if __name__ == "__main__":
    rag = RAGPipeline()
    print("before fun call")
    rag.run("what is the para virtualisation?")
    print("after fun call")