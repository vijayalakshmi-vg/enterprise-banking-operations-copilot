
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "banking_faq_chunks"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Connect to the existing ChromaDB collection.
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(name=COLLECTION_NAME)

# Load the same embedding model used during indexing.
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Test with a customer question.
query = "My money is missing from my account."
print("QUERY BEING EXECUTED:", query)

# Convert the question into an embedding.
query_embedding = model.encode(
    query,
    normalize_embeddings=True
).tolist()

# Retrieve the three most similar FAQ documents.
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3,
    include=["documents", "metadatas", "distances"]
)

print("\nCustomer question:", query)
print("\nTop 3 retrieved FAQs:\n")

for i, document in enumerate(results["documents"][0]):
    metadata = results["metadatas"][0][i]
    distance = results["distances"][0][i]

    print(f"Result {i + 1}")
    print("Intent:", metadata.get("intent", "Old FAQ record"))
    print("Cosine distance:", round(distance, 4))
    print("Document:", document)
    print("-" * 60)
