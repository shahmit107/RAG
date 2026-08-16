import os
import chromadb

# 1. Provide the exact path to your database directory
# (Use absolute paths if running into empty database issues)
DB_PATH = "./my_local_db"
COLLECTION_NAME = "my_documents"

def display_all_records():
    if not os.path.exists(DB_PATH):
        print(f"Error: Directory '{DB_PATH}' does not exist.")
        return

    # Initialize client
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # Check available collections
    collections = [col.name for col in client.list_collections()]
    print(f"Available Collections: {collections}\n")

    if COLLECTION_NAME not in collections:
        print(f"Collection '{COLLECTION_NAME}' not found.")
        return

    # Fetch collection
    collection = client.get_collection(COLLECTION_NAME)
    
    # Retrieve all items (IDs, Documents, Metadatas)
    # Note: Embeddings are excluded by default to keep output readable
    data = collection.get(include=["documents", "metadatas"])
    
    total_records = len(data["ids"])
    print(f"=== Total Records Stored in '{COLLECTION_NAME}': {total_records} ===")
    print("=" * 60)

    if total_records == 0:
        print("No documents found in this collection.")
        return

    for i in range(total_records):
        doc_id = data["ids"][i]
        document = data["documents"][i] if data["documents"] else "None"
        metadata = data["metadatas"][i] if data["metadatas"] else "None"

        print(f"ID:       {doc_id}")
        print(f"Document: {document}")
        print(f"Metadata: {metadata}")
        print("-" * 60)

if __name__ == "__main__":
    display_all_records()
