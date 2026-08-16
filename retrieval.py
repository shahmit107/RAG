from chroma_db import collection
from rank_bm25 import BM25Okapi

# this performs vector search
def query_collection(query_text, embedder, n_results):
    query_embedding = embedder.embed_chunk(query_text)
    results = collection.query( # actual similarity search
        query_embeddings = [query_embedding],
        n_results = n_results
    )
    return results # type = dict

# this performs the keyword search

# first we have to give the index to chunk for tracking them later
def build_bm25_index(chunk_list):
    tokenized = [c.lower().split() for c in chunk_list]
    return BM25Okapi(tokenized) # this returns the weights for the chunks

# this is the main keyword search
def bm25_search(query, bm25_index, chunk_list, n_results=10):
    scores = bm25_index.get_scores(query.lower().split())
    ranked = sorted(zip(chunk_list, scores), key=lambda x: x[1], reverse=True)
    return ranked[:n_results]