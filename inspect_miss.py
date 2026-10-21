
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Load the embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 2. Connect to the existing ChromaDB database
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("banking_faq_chunks")

# 3. Define the question that retrieved the wrong FAQ
query = "The cash never came out but my account was charged"

# 4. Convert the question into an embedding
query_embedding = model.encode([query]).tolist()

# 5. Retrieve the top 3 matching FAQ chunks
results = collection.query(
    query_embeddings=query_embedding,
    n_results=3
)

# 6. Extract the retrieved documents and distances
documents = results["documents"][0]
distances = results["distances"][0]

# 7. Display each result
print(f"Query: {query}\n")
for document, distance in zip(documents, distances):
    print(f"Distance: {distance:.3f}")
    print(document)
    print("-" * 40)
