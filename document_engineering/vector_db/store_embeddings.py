
import json
from pathlib import Path
import chromadb

# Project paths
ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "output" / "embedded_chunks.json"
DB_PATH = ROOT / "output" / "chroma_db"

# Initialize persistent ChromaDB
client = chromadb.PersistentClient(path=str(DB_PATH))

# Create or access collection
collection = client.get_or_create_collection(
    name="banking_policies",
    metadata={"hnsw:space": "cosine"}
)


def store_embeddings():

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for chunk in chunks:

        ids.append(chunk["chunk_id"])
        documents.append(chunk["content"])
        embeddings.append(chunk["embedding"])

        metadata = {
            key: value
            for key, value in chunk.items()
            if key not in ("chunk_id", "content", "embedding")
            and isinstance(value, (str, int, float, bool))
            and value is not None
        }

        metadatas.append(metadata)

    # Insert or update records
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print("Embeddings stored successfully!")
    print("Total records:", collection.count())
    print("Collection:", collection.name)


if __name__ == "__main__":
    store_embeddings()
