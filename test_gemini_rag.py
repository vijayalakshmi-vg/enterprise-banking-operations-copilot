
import os

import chromadb
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer


# 1. Configuration
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "banking_faq_chunks"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-3.8-flash"
TOP_K = 3


def main():
    # 2. Load Gemini API key
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing from .env")

    gemini_client = genai.Client(api_key=api_key)

    # 3. Connect to the existing ChromaDB collection
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection(
        name=COLLECTION_NAME
    )

    # 4. Load the same embedding model used during indexing
    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    # 5. Get a customer question
    question = input("Enter a customer question: ").strip()

    if not question:
        print("Please enter a question.")
        return

    # 6. Convert the question into an embedding
    query_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    # 7. Retrieve the top 3 synthetic FAQ records
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        where={"source_type": "synthetic_demo_content"},
        include=["documents", "metadatas", "distances"]
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not documents:
        print("No FAQ evidence was retrieved. I cannot verify an answer.")
        return

    # 8. Build context from actual retrieved documents
    context_parts = []

    for i, document in enumerate(documents):
        intent = metadatas[i].get("intent", "unknown")
        distance = distances[i]

        context_parts.append(
            f"Retrieved document {i + 1}\n"
            f"Intent: {intent}\n"
            f"Cosine distance: {distance:.4f}\n"
            f"Content:\n{document}"
        )

    retrieved_context = "\n\n".join(context_parts)

    print("\nRetrieved FAQ documents:")
    for i, metadata in enumerate(metadatas):
        print(
            f"{i + 1}. Intent: "
            f"{metadata.get('intent', 'unknown')} | "
            f"Distance: {distances[i]:.4f}"
        )

    
    # 9. Compare Gemini with and without retrieved context

    base_instructions = """
You are an Enterprise Banking Operations Copilot.

- Provide information and guidance only.
- Never perform or claim to perform customer account actions.
- Never invent bank policies, fees, deadlines, or guarantees.
- Never request passwords, PINs, or full account credentials.
- If the available information is insufficient, say so.
- Recommend official bank support when account-specific verification
  is needed.
- The supplied FAQ records are synthetic demo content, not verified
  bank policies.
- Do not present unsupported assumptions as facts.
"""

    # Test A: Include the actual ChromaDB results.
    prompt_with_context = f"""
{base_instructions}

Use the retrieved documents as evidence. If they do not support
a specific answer, explicitly acknowledge that limitation.

Retrieved context:
{retrieved_context}

Customer question:
{question}

Provide a concise response.
"""

    # Test B: No retrieved documents.
    prompt_without_context = f"""
{base_instructions}

No company FAQ documents were provided. Do not pretend to have
consulted internal documents.

Customer question:
{question}

Provide a concise response.
"""

    response_a = gemini_client.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=prompt_with_context
    )

    response_b = gemini_client.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=prompt_without_context
    )

    print("\n========== TEST A: WITH RETRIEVAL ==========\n")
    print(response_a.text or "No text response returned.")

    print("\n========== TEST B: WITHOUT RETRIEVAL ==========\n")
    print(response_b.text or "No text response returned.")

if __name__ == "__main__":
    main()