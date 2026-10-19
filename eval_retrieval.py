
import chromadb
from sentence_transformers import SentenceTransformer

# Load the same embedding model used by our RAG pipeline
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load the persisted FAQ collection
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("banking_faq_chunks")

# Each case: (customer question, expected FAQ)
# None means the knowledge base does not cover the question.
CASES = [
    # FAQ 1: Pending ATM withdrawal
    ("My ATM withdrawal is still pending", "FAQ 1"),
    ("The cash never came out but my account was charged", "FAQ 1"),
    ("When will I get my money back for a failed ATM withdrawal?", "FAQ 1"),

    # FAQ 2: Card declined
    ("Why was my card declined?", "FAQ 2"),
    ("My debit card payment keeps getting rejected", "FAQ 2"),
    ("Why can't I use my card to pay?", "FAQ 2"),

    # FAQ 3: Lost or stolen card
    ("I lost my card", "FAQ 3"),
    ("Someone stole my bank card", "FAQ 3"),
    ("How can I report a missing card?", "FAQ 3"),

    # FAQ 4: International transaction fees
    ("Do I pay fees when I use my card abroad?", "FAQ 4"),
    ("What extra charges apply to overseas purchases?", "FAQ 4"),
    ("Will I be charged for using my card internationally?", "FAQ 4"),

    # FAQ 5: Unrecognized cash withdrawal
    ("There is a cash withdrawal on my statement I don't recognize", "FAQ 5"),
    ("Someone may have withdrawn money from my account", "FAQ 5"),
    ("I see a suspicious ATM transaction", "FAQ 5"),

    # Out-of-scope questions
    ("How do I open a savings account?", None),
    ("What is my daily ATM withdrawal limit?", None),
    ("How can I apply for a home loan?", None),
    ("What is the current interest rate on savings?", None),
    ("How do I change my account address?", None),
]

# 4. Initialize counters and distance lists
n_in = 0
in_top1 = 0
in_top3 = 0
in_distances = []
out_distances = []

# 5. Evaluate each question
for query, expected in CASES:
    # Convert the question into a 384-dimensional embedding
    query_embedding = model.encode([query]).tolist()

    # Retrieve the three closest FAQ chunks
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    # Extract retrieved FAQ texts and their distances
    documents = results["documents"][0]
    distances = results["distances"][0]

    # Extract FAQ IDs, such as "FAQ 3", from each document
    retrieved_faqs = [doc.split(":")[0] for doc in documents]

    # Handle questions not covered by our knowledge base
    if expected is None:
        out_distances.append(distances[0])
        print(f"OUT   {distances[0]:.3f}  {query}")
        continue

    # Count an in-scope question and record its top distance
    n_in += 1
    in_distances.append(distances[0])

    # Check whether the expected FAQ ranks first
    top1_correct = retrieved_faqs[0] == expected
    in_top1 += top1_correct

    # Check whether the expected FAQ appears in the top three
    top3_correct = expected in retrieved_faqs
    in_top3 += top3_correct

    # Print the retrieval result
    status = "OK  " if top1_correct else "MISS"
    print(
        f"{status}  {distances[0]:.3f}  {query}"
        f"  -> Top result: {retrieved_faqs[0]}"
    )

# 6. Display evaluation metrics
print("\n--- Evaluation Summary ---")
print(f"In-scope questions: {n_in}")
print(f"Top-1 accuracy: {in_top1}/{n_in} = {in_top1 / n_in:.1%}")
print(f"Top-3 recall:   {in_top3}/{n_in} = {in_top3 / n_in:.1%}")

# 7. Compare the top-result distance ranges
print(
    f"In-scope top distance: "
    f"{min(in_distances):.3f} - {max(in_distances):.3f}"
)
print(
    f"Out-of-scope top distance: "
    f"{min(out_distances):.3f} - {max(out_distances):.3f}"
)
