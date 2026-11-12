
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai

# Load environment variables
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Connect to ChromaDB
DB_PATH = PROJECT_ROOT / "document_engineering" / "output" / "chroma_db"

chroma_client = chromadb.PersistentClient(path=str(DB_PATH))
collection = chroma_client.get_collection(
    name="banking_policies"
)


def generate_answer(question, top_k=3):

    # Generate query embedding
    embedding_response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=question
    )

    query_embedding = embedding_response.embeddings[0].values

    # Retrieve relevant policy chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    context = "\n\n".join(results["documents"][0])

    prompt = f"""
You are a banking operations assistant.

Answer the user's question using only the provided policy context.

Rules:
1. Do not invent banking procedures.
2. If the context does not contain the answer, say that the policy
   context does not provide enough information.
3. Do not ask customers for their complete PIN or password.
4. Keep the answer clear and actionable.

Policy Context:
{context}

User Question:
{question}

Answer:
"""

    # Generate grounded answer
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt
    )

    print("\nUser Question:")
    print(question)

    print("\nRetrieved Context:")
    print(context)

    print("\nGenerated Answer:")
    print(response.text)


if __name__ == "__main__":

    question = "How can I block my debit card if it is stolen?"

    generate_answer(question)
