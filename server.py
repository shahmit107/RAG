from fastapi import FastAPI
from query import RAGPipeline
app = FastAPI()

rag = RAGPipeline()
@app.post("/query")
def query_rag(query_text: str):
    answer = rag.run(query_text)
    return {
        "answer": answer
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)