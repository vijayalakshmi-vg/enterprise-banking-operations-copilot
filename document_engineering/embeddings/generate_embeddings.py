
import json
from pathlib import Path
from google import genai
from dotenv import load_dotenv

# Project paths
ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "output" / "chunks_v2.json"
OUTPUT_FILE = ROOT / "output" / "embedded_chunks.json"

# Load Gemini API key from .env
load_dotenv()

client = genai.Client()

MODEL_NAME = "gemini-embedding-001"


def generate_embeddings(chunks):

    embedded_chunks = []

    # Process chunks in batches
    batch_size = 32

    for start in range(0, len(chunks), batch_size):

        batch = chunks[start:start + batch_size]

        texts = [chunk["content"] for chunk in batch]

        response = client.models.embed_content(
            model=MODEL_NAME,
            contents=texts
        )

        for chunk, embedding in zip(batch, response.embeddings):

            embedded_chunks.append({
                **chunk,
                "embedding": embedding.values
            })

        print(
            f"Processed {len(embedded_chunks)} / {len(chunks)} chunks"
        )

    return embedded_chunks


def main():

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    embedded_chunks = generate_embeddings(chunks)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            embedded_chunks,
            file,
            indent=4
        )

    print("\nEmbedding generation completed!")
    print("Total embedded chunks:", len(embedded_chunks))
    print("Vector dimensions:", len(embedded_chunks[0]["embedding"]))
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
