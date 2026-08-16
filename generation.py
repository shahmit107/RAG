from google import genai
import os
from dotenv import load_dotenv

GENERATION_MODEL = "___"
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client()

def build_prompt(query_text, retrieved_chunks):
    """
    query_text: the user's original question
    retrieved_chunks: list of chunk text strings (from your query_collection results) means context
    """
    # Step 1: join the retrieved chunks into one context block
    context = "\n\n".join(retrieved_chunks)
    #method to stitch them together, separated by double newlines so the LLM can easily distinguish between different documents.

    # Step 2: build the final prompt string using an f-string, following the pattern shown above
    prompt = f"""Answer the question using ONLY the context below. If the answer isn't in the context, say you don't know. Analyze deeply and don't go off-topic.
    Context: {context}
    Question: {query_text}"""

    return prompt