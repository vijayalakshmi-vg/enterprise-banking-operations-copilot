
# evaluate_retrieval.py

# Each test contains a customer question and its expected FAQ intent.
    
test_cases = [
    # Original six cases
    {
        "query": "Why hasn't my refund arrived?",
        "expected_intent": "Refund_not_showing_up"
    },
    {
        "query": "The merchant processed my refund, but I still don't see the money.",
        "expected_intent": "Refund_not_showing_up"
    },
    {
        "query": "I don't recognize a card payment on my statement.",
        "expected_intent": "card_payment_not_recognised"
    },
    {
        "query": "I sent money to another account, but the transfer is still pending.",
        "expected_intent": "pending_transfer"
    },
    {
        "query": "My bank transfer failed. What should I do?",
        "expected_intent": "failed_transfer"
    },
    {
        "query": "How do I activate my new card?",
        "expected_intent": "activate_my_card"
    },

    # Six additional cases
    {
        "query": "My card has arrived, but I cannot use it.",
        "expected_intent": "card_not_working"
    },
    {
        "query": "The shop says my payment went through, but my bank shows it as pending.",
        "expected_intent": "pending_card_payment"
    },
    {
        "query": "I was charged twice for the same purchase.",
        "expected_intent": "transaction_charged_twice"
    },
    {
        "query": "How can I find out why my bank transfer was rejected?",
        "expected_intent": "declined_transfer"
    },
    {
        "query": "I need to freeze my card because I lost it.",
        "expected_intent": "lost_or_stolen_card"
    },
    {
        "query": "My contactless payment is not working.",
        "expected_intent": "contactless_not_working"
    },
    
    {
        "query": "My transfer was rejected by the bank.",
        "expected_intent": "declined_transfer"
    },
    {
        "query": "My transfer failed and the money has not arrived back.",
        "expected_intent": "failed_transfer"
    },
    {
        "query": "My card works, but contactless payments keep failing.",
        "expected_intent": "contactless_not_working"
    },
    {
        "query": "I want to get my money back for a card purchase.",
        "expected_intent": "request_refund"
    },


]

print("Number of evaluation queries:", len(test_cases))


import chromadb
from sentence_transformers import SentenceTransformer

# Connect to the existing ChromaDB database
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "banking_faq_chunks"

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(name=COLLECTION_NAME)

# Load the same embedding model used by the RAG pipeline
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("ChromaDB connected successfully!")
print("Total records in collection:", collection.count())


top1_correct = 0
top3_correct = 0
evaluation_results = []
for test in test_cases:
    query = test["query"]
    expected_intent = test["expected_intent"]

    # Convert the customer question into an embedding
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    # Retrieve the three most similar FAQs
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        where={"source_type": "synthetic_demo_content"},
        include=["metadatas", "distances"]
    )

    
    retrieved_intents = [
        metadata.get("intent", "unknown")
        if metadata is not None
        else "unknown"
        for metadata in results["metadatas"][0]
    ]
    evaluation_results.append({
    "query": query,
    "expected_intent": expected_intent,
    "top1_intent": retrieved_intents[0],
    "top1_correct": retrieved_intents[0] == expected_intent,
    "top3_intents": ", ".join(retrieved_intents),
    "top3_correct": expected_intent in retrieved_intents
    })


    # Check whether the expected intent ranks first
    if retrieved_intents[0] == expected_intent:
        top1_correct += 1

    # Check whether the expected intent appears in the top three
    if expected_intent in retrieved_intents:
        top3_correct += 1

    print("\nCustomer question:", query)
    print("Expected intent:", expected_intent)
    print("Retrieved intents:", retrieved_intents)
    print("Top-1 correct:", retrieved_intents[0] == expected_intent)
    print("Top-3 correct:", expected_intent in retrieved_intents)
    print("-" * 60)

print("\n--- Retrieval Evaluation Summary ---")
print(f"Top-1 accuracy: {top1_correct}/{len(test_cases)} "
      f"({top1_correct / len(test_cases):.1%})")
print(f"Recall@3: {top3_correct}/{len(test_cases)} "
      f"({top3_correct / len(test_cases):.1%})")



import csv
from pathlib import Path

output_path = Path("evaluation_results/retrieval_evaluation.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "query",
            "expected_intent",
            "top1_intent",
            "top1_correct",
            "top3_intents",
            "top3_correct",
        ],
    )
    writer.writeheader()
    writer.writerows(evaluation_results)

print(f"\nEvaluation results saved to: {output_path}")
