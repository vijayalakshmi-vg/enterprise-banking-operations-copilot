
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai
from document_engineering.retrieval.retrieve_documents import retrieve_documents


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "document_engineering"
    / "output"
    / "policy_embeddings.json"
)

DB_PATH = (
    PROJECT_ROOT
    / "document_engineering"
    / "output"
    / "chroma_db"
)

COLLECTION_NAME = "banking_policy_catalog"
MODEL_NAME = "gemini-embedding-001"
# ==========================================
# 2. INITIALIZE CLIENTS
# ==========================================

load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

genai_client = genai.Client(api_key=api_key)

chroma_client = chromadb.PersistentClient(
    path=str(DB_PATH)
)


# ==========================================
# 3. LOAD POLICY EMBEDDINGS
# ==========================================

def load_policy_embeddings():

    import json

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        policies = json.load(file)

    return policies


# ==========================================
# 4. CREATE POLICY COLLECTION
# ==========================================

def create_policy_collection():

    policies = load_policy_embeddings()

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    collection.upsert(
        ids=[
            policy["document_id"]
            for policy in policies
        ],

        embeddings=[
            policy["embedding"]
            for policy in policies
        ],

        documents=[
            policy["description"]
            for policy in policies
        ],

        metadatas=[
            {"document_id": policy["document_id"]}
            for policy in policies
        ]
    )

    print(
        f"Policy collection ready: {collection.count()} policies"
    )

    return collection


# ==========================================
# 5. GENERATE QUESTION EMBEDDING
# ==========================================

def generate_query_embedding(question):

    response = genai_client.models.embed_content(
        model=MODEL_NAME,
        contents=question
    )

    return response.embeddings[0].values


# ==========================================
# 6. POLICY ROUTER
# ==========================================

def route_question(question, top_k=3):

    collection = chroma_client.get_collection(
        name=COLLECTION_NAME
    )

    query_embedding = generate_query_embedding(question)
   

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    print("\nUser Question:", question)
    print("\nTop Matching Policies:\n")

    for i in range(len(results["ids"][0])):

        distance = results["distances"][0][i]

        similarity = 1 - distance

        print("Policy ID:", results["ids"][0][i])

        print(
            "Cosine Similarity:",
            round(similarity, 4)
        )

        print(
            "Description:",
            results["documents"][0][i]
        )

        print("-" * 60)

    return results

def select_relevant_policies(results, threshold=0.70):

    selected_policies = []

    policy_ids = results["ids"][0]
    distances = results["distances"][0]
    descriptions = results["documents"][0]

    for policy_id, distance, description in zip(
        policy_ids, distances, descriptions
    ):

        similarity = 1 - distance

        if similarity >= threshold:

            selected_policies.append({
                "document_id": policy_id,
                "description": description,
                "similarity": round(similarity, 4)
            })

    # Fallback: always retain the highest-ranked policy
    # if no policy meets the threshold.
    if not selected_policies and policy_ids:

        selected_policies.append({
            "document_id": policy_ids[0],
            "description": descriptions[0],
            "similarity": round(1 - distances[0], 4)
        })

    return selected_policies

# ==========================================
# 7. POLICY ROUTING TEST DATASET
# ==========================================

test_cases = [

    {
        "question": "How can a customer update their residential address?",
        "expected_policy": "POL-CIM-001"
    },

    {
        "question": "Why would a funds transfer fail and how can it be reversed?",
        "expected_policy": "POL-FTO-002"
    },

    {
        "question": "How can I block and replace my stolen debit card?",
        "expected_policy": "POL-CARD-003"
    },

    {
        "question": "What should a customer do after detecting suspicious account activity?",
        "expected_policy": "POL-SEC-004"
    },

    {
        "question": "What identity documents and checks are required to open a bank account?",
        "expected_policy": "POL-KYC-005"
    },

    {
        "question": "What eligibility criteria must a customer meet to apply for a loan?",
        "expected_policy": "POL-LOAN-006"
    },

    {
        "question": "What service fees can a bank charge its customers?",
        "expected_policy": "POL-FEE-007"
    },

    {
        "question": "How does a customer dispute an unauthorized card transaction?",
        "expected_policy": "POL-DIS-008"
    },

    {
        "question": "How can a customer submit and escalate a banking complaint?",
        "expected_policy": "POL-COM-009"
    },

    {
        "question": "How can a customer reactivate a dormant bank account?",
        "expected_policy": "POL-ACD-010"
    }
]


# ==========================================
# 8. EVALUATE POLICY ROUTING
# ==========================================

def evaluate_policy_router():

    top_k = 3

    total_questions = len(test_cases)

    top1_correct = 0
    hit_at_k = 0

    print("\n")
    print("=" * 75)
    print("POLICY ROUTER EVALUATION")
    print("=" * 75)

    for index, case in enumerate(test_cases, start=1):

        question = case["question"]

        expected_policy = case["expected_policy"]

        print(f"\nTest Question {index}/{total_questions}")

        results = route_question(
            question,
            top_k=top_k
        )

        retrieved_ids = results["ids"][0]
        # Select policies based on similarity threshold
        selected_policies = select_relevant_policies(
            results,
            threshold=0.70
        )
        selected_policy_ids = [
            policy["document_id"]
            for policy in selected_policies
        ]

        print("\nSelected Policy IDs:", selected_policy_ids)

        retrieved_chunks = retrieve_documents(
            question,
            selected_policy_ids,
            top_k=5
        )

        selected_ids = [
            policy["document_id"]
            for policy in selected_policies
        ]

        print("\nSelected Policies:", selected_ids)

        for policy in selected_policies:
            print(
                policy["document_id"],
                "Similarity:",
                policy["similarity"]
            )

        # Top-1 accuracy
        if retrieved_ids[0] == expected_policy:
            top1_correct += 1

        # Hit Rate@K
        if expected_policy in retrieved_ids:
            hit_at_k += 1

        print("\nExpected Policy:", expected_policy)

        print("Retrieved Policies:", retrieved_ids)

        print(
            "Rank 1 Correct:",
            retrieved_ids[0] == expected_policy
        )

        print(
            f"Hit@{top_k}:",
            expected_policy in retrieved_ids
        )

        print("=" * 75)

    # Calculate metrics

    top1_accuracy = top1_correct / total_questions

    hit_rate = hit_at_k / total_questions

    print("\n")
    print("=" * 75)
    print("FINAL EVALUATION RESULTS")
    print("=" * 75)

    print(
        f"Total Questions: {total_questions}"
    )

    print(
        f"Top-1 Accuracy: {top1_correct}/{total_questions} "
        f"({top1_accuracy:.2%})"
    )

    print(
        f"Hit Rate@{top_k}: {hit_at_k}/{total_questions} "
        f"({hit_rate:.2%})"
    )

    print("=" * 75)


# ==========================================
# 9. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    create_policy_collection()

    evaluate_policy_router()
