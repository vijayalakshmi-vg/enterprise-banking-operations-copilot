import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")
col = client.get_collection("banking_faq_chunks")
print("Stored chunks:", col.count(), flush=True)

queries = [
    "I lost my card",
    "my ATM withdrawal is still pending",
    "do I pay fees when I use my card abroad",
    "how do I open a savings account",
]

for q in queries:
    res = col.query(query_embeddings=model.encode([q]).tolist(), n_results=3)
    print("\nQUERY:", q)
    for doc, dist in zip(res["documents"][0], res["distances"][0]):
        print(f"  {dist:.3f}  {doc[:90]!r}")