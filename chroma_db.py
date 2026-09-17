import chromadb

# Creates the './my_local_db' directory automatically if it doesn't exist
client = chromadb.PersistentClient(path="./my_local_db")

# Safer method: Fetches 'my_documents' if it exists, or creates it if it's new
collection = client.get_or_create_collection(name="my_documents")

def add_chunks_to_collection(chunks, embeddings):
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    documents = [chunk.page_content for chunk in chunks]
    metadatas = []
    for chunk in chunks:
        page = chunk.metadata.get("page", None)
        metadatas.append({
            "source": chunk.metadata.get("source", "unknown"), # .get(key, default_value)
            "page": (page + 1) if page is not None else -1,
        })

    collection.add(
        ids = ids,
        embeddings = embeddings,
        documents = documents,
        metadatas = metadatas
    )
    print(f"Added {len(chunks)} chunks to collection '{collection}'")

if __name__ == "__main__":
    print("Collections in DB:", client.list_collections())
    print("Total documents stored:", collection.count())

    if collection.count() > 0:
        print(collection.peek(5))
    else:
        print("No documents in collection yet — nothing added so far.")