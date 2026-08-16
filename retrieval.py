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

# retrieves the chunk stored in the db
def get_all_chunks_from_db():
    data = collection.get()
    return data['documents']

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


def merge_rrf(vector_results, bm25_results, k=60):
    scores = {}  # chunk_text -> combined RRF score

    # Loop 1: go through vector_results
    for rank, chunk in enumerate(vector_results):
        # add 1/(k + rank + 1) to scores[chunk]
        scores[chunk] = scores.get(chunk, 0) + 1 / (k + rank + 1)
        # use scores.get(chunk, 0) + ... so it doesn't crash on a new chunk

    # Loop 2: go through bm25_results — remember these are (chunk, score) tuples
    for rank, (chunk, _) in enumerate(bm25_results):
        # same idea — add 1/(k + rank + 1) to scores[chunk]
        scores[chunk] = scores.get(chunk, 0) + 1 / (k + rank + 1)
        # if this chunk was already scored in Loop 1, this should ADD to it, not overwrite

    # sort `scores` by value, descending
    ranked_pairs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    ranked_texts = [doc for doc, score in ranked_pairs]
    return ranked_texts

# return just the chunk texts (not the scores) as a plain list, best first