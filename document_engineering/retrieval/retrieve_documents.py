
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai

# Load Gemini API key
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Connect to existing ChromaDB
DB_PATH = Path(__file__).resolve().parents[1] / "output" / "chroma_db"

chroma_client = chromadb.PersistentClient(path=str(DB_PATH))

collection = chroma_client.get_collection(
    name="banking_policies"
)


def retrieve_documents(question, top_k=3):

    # Convert user question into an embedding
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=question
    )

    query_embedding = response.embeddings[0].values

    # Retrieve similar chunks from ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    print("\nUser Question:")
    print(question)

    print("\nRetrieved Documents:")

    for i in range(len(results["documents"][0])):
        print(f"\nResult {i + 1}")
        print("Content:", results["documents"][0][i])
        print("Metadata:", results["metadatas"][0][i])
        print("Cosine Distance:", results["distances"][0][i])


if __name__ == "__main__":

    question = "How can I block my debit card if it is stolen?"

    retrieve_documents(question, top_k=3)
