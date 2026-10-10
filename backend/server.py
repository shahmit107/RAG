import logging
from pathlib import Path
from threading import Lock
from uuid import uuid4

from backend.paths import UPLOADS_DIR

from fastapi import Body, FastAPI, HTTPException

from backend.pipeline.chroma_db import collection
from backend.pipeline.embedding_manager import EmbeddingManager
from backend.pipeline.retrieval import build_bm25_index, get_all_chunks_from_db

app = FastAPI()
logger = logging.getLogger(__name__)
STORAGE_DIR = UPLOADS_DIR
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
pipeline_lock = Lock()
rag = None


def process_upload(path, filename):
    from backend.pipeline.chunking import CHUNK_OVERLAP, CHUNK_SIZE, split_documents
    from langchain_community.document_loaders import PyMuPDFLoader, TextLoader

    ids = []
    try:
        if path.suffix == ".pdf":
            loader = PyMuPDFLoader(str(path), extract_tables="markdown", extract_images=True)
        else:
            loader = TextLoader(str(path), encoding="utf-8")
        documents = loader.load()
        chunks = split_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)
        chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
        if not chunks:
            raise ValueError("No readable text was found")

        ids = [f"upload_{path.stem}_{i}" for i in range(len(chunks))]
        embedder = EmbeddingManager()
        for start in range(0, len(chunks), 100):
            batch = chunks[start:start + 100]
            texts = [chunk.page_content for chunk in batch]
            metadata = []
            for chunk in batch:
                page = chunk.metadata.get("page")
                metadata.append({"source": filename, "page": page + 1 if page is not None else -1})
            embeddings = embedder.embed_chunks_batch(texts)
            collection.add(
                ids=ids[start:start + len(batch)],
                documents=texts, embeddings=embeddings, metadatas=metadata,
            )

        if rag is not None:
            all_chunks = get_all_chunks_from_db()
            bm25_index = build_bm25_index(all_chunks)
            rag.all_chunks = all_chunks
            rag.bm25_index = bm25_index
        return len(chunks)
    except Exception as exc:
        logger.exception("Could not process uploaded document")
        # Delete only this upload's vectors, leaving existing documents intact.
        for start in range(0, len(ids), 100):
            collection.delete(ids=ids[start:start + 100])
        raise HTTPException(422, "The file was saved, but processing failed. Check the file and try again.") from exc


@app.post("/upload", status_code=201)
def upload_file(filename: str, content: bytes = Body(default=b"", media_type="application/octet-stream")):
    filename = filename.replace("\\", "/").rsplit("/", 1)[-1]
    extension = Path(filename).suffix.lower()
    if extension not in {".pdf", ".txt"}:
        raise HTTPException(415, "Please upload a PDF or UTF-8 text file.")

    if not content:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "The file must be 25 MB or smaller.")
    if extension == ".pdf" and not content.startswith(b"%PDF-"):
        raise HTTPException(415, "This file does not appear to be a PDF.")
    if extension == ".txt":
        try:
            content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(415, "Text files must use UTF-8 encoding.") from exc
        if b"\x00" in content:
            raise HTTPException(415, "This text file contains binary data.")

    with pipeline_lock:
        document_id = str(uuid4())
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        path = STORAGE_DIR / f"{document_id}{extension}"
        with path.open("xb") as target:
            target.write(content)
        count = process_upload(path, filename)
    return {"document_id": document_id, "filename": filename, "chunks": count}


@app.post("/query")
def query_rag(query_text: str):
    global rag

    query_text = query_text.strip()
    if not query_text:
        raise HTTPException(400, "Please enter a question.")
    # Queries wait for an upload to finish instead of seeing partially added chunks.
    with pipeline_lock:
        if collection.count() == 0:
            raise HTTPException(400, "Upload a document before asking a question.")
        if rag is None:
            from backend.pipeline.query import RAGPipeline

            rag = RAGPipeline()
        answer = rag.run(query_text)
    return {"answer": answer}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
