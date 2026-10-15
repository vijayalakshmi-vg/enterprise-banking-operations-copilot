import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
import chromadb
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "banking_faq_chunks"

#define document path
Document_path= Path("documents/banking_faq.txt")

def load_document(file_path):
    with open(file_path,"r", encoding="utf-8") as file:
        text = file.read()

    return text

def split_into_chunks(text):
    """Split the FAQ text into one chunk per FAQ (heading + answer together)."""
    sections = re.split(r"(?m)^(?=FAQ \d+:)", text)
    chunks = [
        re.sub(r"\n\s*\n", "\n", section.strip())
        for section in sections
        if section.strip()
    ]
    return chunks

def generate_embeddings(chunks):
    """Convert text chunks into numerical vectors."""
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    embeddings = embedding_model.encode(
        chunks,
        show_progress_bar=True
    )

    return embedding_model, embeddings

##stroing in chromadb

def store_chunks(chunks, embeddings):
    print("1. Creating ChromaDB client...", flush=True)

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    print("2. ChromaDB client created.", flush=True)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    print("3. Collection ready.", flush=True)

    
    print("3a. Preparing IDs, documents, and embeddings...", flush=True)

    ids = [
        f"chunk_{i:03d}"
        for i in range(1, len(chunks) + 1)
    ]

    documents = chunks
    embedding_vectors = embeddings.tolist()

    print("Number of IDs:", len(ids), flush=True)
    print("Number of documents:", len(documents), flush=True)
    print("Number of vectors:", len(embedding_vectors), flush=True)
    print("First vector dimensions:", len(embedding_vectors[0]), flush=True)

    print("3b. Starting upsert...", flush=True)

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embedding_vectors
    )

    print("4. Upsert completed.", flush=True)
    print("Total stored chunks:", collection.count(), flush=True)


if __name__=="__main__":
    document_text=load_document(Document_path)
    chunks=split_into_chunks(document_text)

    print(f"Document loaded successfully")
    print(f"Number of chunks:{len(chunks)}")
    print(chunks)

    for index, chunk in enumerate(chunks, start=1):
        print(f"\n--- Chunk {index} ---")
        print(chunk)


    embedding_model, embeddings = generate_embeddings(chunks)

    print("\nEmbedding test successful.")
    print("Number of chunks:", len(chunks))
    print("Number of embeddings:", len(embeddings))
    print("First embedding dimensions:", embeddings[0].shape)   
    print("Starting ChromaDB storage...")
    collection = store_chunks(chunks, embeddings)
    print(collection)   
    

            
            

