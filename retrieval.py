# contains what should be performed when the fun is being called
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
    result = []
    for document, metadata in zip( # zip() pairs each document with its corresponding metadata
            data['documents'],
            data['metadatas']
    ):
        result.append({
            "document": document,
            "metadata": metadata
        })

    return result
# the type of the result:
# [
#     {d1,m1},
#     {d2,m2}.....
# ]

# this performs the keyword search
# first we have to give the index to chunk for tracking them later
def build_bm25_index(chunk_list):
    tokenized = [c["document"].lower().split() for c in chunk_list]
    return BM25Okapi(tokenized) # this returns the weights for the chunks

# this is the main keyword search
def bm25_search(query, bm25_index, chunk_list, n_results=10):
    scores = bm25_index.get_scores(query.lower().split())
    ranked = sorted(zip(chunk_list, scores), key=lambda x: x[1], reverse=True)
    return ranked[:n_results]

def merge_rrf(vector_results, bm25_results, k=60):
    scores = {}  # chunk_text -> combined RRF score
    vect_retainer = {}
    bm25_retainer = {}
    ranked_chunks = []

    # Loop 1: go through vector_results
    for rank, chunk in enumerate(vector_results):
        # add 1/(k + rank + 1) to scores[chunk]
        scores[chunk["document"]] = scores.get(chunk["document"], 0) + 1 / (k + rank + 1)
        vect_retainer[chunk["document"]] = chunk
        # use scores.get(chunk, 0) + ... so it doesn't crash on a new chunk

    # Loop 2: go through bm25_results — remember these are (chunk, score) tuples
    for rank, (chunk, _) in enumerate(bm25_results):
        # same idea — add 1/(k + rank + 1) to scores[chunk]
        scores[chunk["document"]] = scores.get(chunk["document"], 0) + 1 / (k + rank + 1)
        bm25_retainer[chunk["document"]] = chunk
        # if this chunk was already scored in Loop 1, this should ADD to it, not overwrite

    # sort `scores` by value, descending
    ranked_pairs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    for doc, score in ranked_pairs:
        if(vect_retainer.get(doc)):
            ranked_chunks.append(vect_retainer.get(doc))
        elif(bm25_retainer.get(doc)):
            ranked_chunks.append(bm25_retainer.get(doc))

    return ranked_chunks

# return just the chunk texts (not the scores) as a plain list, best first

def rerank(query, chunks, cross_encoder, top_n ):

    print("this is from the rerank function")
    """
    query: the user's question (string)
    chunks: list of chunk text strings (your merge_rrf output, e.g. top 10)
    top_n: how many chunks to keep after reranking

    Returns:
        List of chunk texts, re-ordered by cross-encoder relevance, top_n only.
    """
    # build pairs, a list of [query, chunk] for every chunk
    pairs = [(query, chunk["document"]) for chunk in chunks]

    # get scores, pass pairs into cross_encoder.predict()
    scores = cross_encoder.predict(pairs)

    # sort chunks by score, descending
    # hint: same pattern as merge_rrf's sorted(..., key=..., reverse=True) —
    # but here you're sorting (chunk, score) pairs, not a dict
    ranked_pairs = sorted(
        zip(chunks, scores),
        key = lambda x: x[1],
        reverse=True)


    # return just the top_n chunk texts (no scores)
    return [chunk for chunk, score in ranked_pairs[:top_n]]

if __name__ == "__main__":
    print(get_all_chunks_from_db())