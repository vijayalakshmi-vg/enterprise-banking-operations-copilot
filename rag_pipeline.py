
from pathlib import Path

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "banking_faq_chunks"

DOCUMENT_PATH = Path("documents/banking77_rag_faq_starter (1).csv")


# --------------------------------------------------
# 2. Load and validate the FAQ CSV
# --------------------------------------------------

def load_faqs(file_path):
    """Load FAQs and validate the required CSV columns."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"FAQ CSV not found: {file_path.resolve()}"
        )

    df = pd.read_csv(file_path)

    required_columns = {"intent", "faq_question", "answer"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing CSV columns: {missing_columns}"
        )

    # Check for missing or empty values.
    for column in required_columns:
        df[column] = df[column].astype("string").str.strip()

        if df[column].isna().any() or df[column].eq("").any():
            raise ValueError(
                f"Column '{column}' contains missing or empty values."
            )

    # This starter dataset should contain one FAQ per intent.
    if df["intent"].duplicated().any():
        duplicates = df.loc[
            df["intent"].duplicated(keep=False), "intent"
        ].tolist()

        raise ValueError(f"Duplicate intents found: {duplicates}")

    return df


# --------------------------------------------------
# 3. Prepare documents, metadata, and IDs
# --------------------------------------------------

def prepare_documents(df):
    """Create one searchable document for each FAQ."""

    documents = []
    metadatas = []
    ids = []

    for _, row in df.iterrows():

        intent = str(row["intent"])
        question = str(row["faq_question"])
        answer = str(row["answer"])

        # Combine the intent, question, and answer.
        document = (
            f"Intent: {intent}\n"
            f"Question: {question}\n"
            f"Answer: {answer}"
        )

        documents.append(document)

        # Metadata is stored separately from document text.
        metadatas.append({
            "intent": intent,
            "source_type": "synthetic_demo_content"
        })

        # Use stable IDs so rerunning the pipeline updates
        # existing FAQ records instead of creating duplicates.
        ids.append(f"faq_{intent}")

    return documents, metadatas, ids


# --------------------------------------------------
# 4. Generate embeddings
# --------------------------------------------------

def generate_embeddings(documents):
    """Convert FAQ documents into numerical vectors."""

    print("\nLoading embedding model...", flush=True)

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    print("Generating embeddings...", flush=True)

    embeddings = embedding_model.encode(
        documents,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    return embeddings


# --------------------------------------------------
# 5. Store FAQs in ChromaDB
# --------------------------------------------------

def store_faqs(documents, metadatas, ids, embeddings):
    """Store or update FAQ records in ChromaDB."""

    print("\nConnecting to ChromaDB...", flush=True)

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    print("Collection ready:", collection.name)

    print("Upserting FAQ records...", flush=True)

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings.tolist()
    )

    print("\nIndexing completed successfully.")
    print("FAQs processed:", len(documents))
    print("Embedding dimensions:", len(embeddings[0]))
    print("Total records in collection:", collection.count())

    return collection


# --------------------------------------------------
# 6. Main execution
# --------------------------------------------------

if __name__ == "__main__":

    # Step 1: Load FAQs.
    df = load_faqs(DOCUMENT_PATH)

    print("FAQ rows loaded:", len(df))
    print("Unique intents:", df["intent"].nunique())

    # Step 2: Prepare searchable documents and metadata.
    documents, metadatas, ids = prepare_documents(df)

    print("Documents prepared:", len(documents))
    print("Metadata records prepared:", len(metadatas))

    print("\nFirst document preview:\n")
    print(documents[0])

    print("\nFirst metadata preview:")
    print(metadatas[0])

    # Step 3: Generate embeddings.
    embeddings = generate_embeddings(documents)

    print("\nNumber of embeddings:", len(embeddings))
    print("Dimensions per embedding:", embeddings.shape[1])

    
    # Step 4: Store records in ChromaDB.
    collection = store_faqs(
    documents,
    metadatas,
    ids,
    embeddings
)

print("\nFinal collection count:", collection.count())