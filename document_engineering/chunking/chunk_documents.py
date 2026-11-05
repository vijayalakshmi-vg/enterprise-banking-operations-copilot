
import json
from pathlib import Path
import tiktoken

# Project paths
ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "output" / "cleaned_records.json"
OUTPUT_FILE = ROOT / "output" / "chunks.json"

# Chunking configuration
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text):
    return len(tokenizer.encode(text))


def split_long_text(text):
    tokens = tokenizer.encode(text)
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP

    for start in range(0, len(tokens), step):
        part = tokens[start:start + CHUNK_SIZE]

        if not part:
            break

        chunks.append(tokenizer.decode(part))

        if start + CHUNK_SIZE >= len(tokens):
            break

    return chunks


def create_chunks(records):
    chunks = []
    current_paragraphs = []
    current_metadata = None

    def save_chunk():
        if not current_paragraphs:
            return

        text = "\n\n".join(current_paragraphs)

        for part in split_long_text(text):
            chunks.append({
                "chunk_id": f"chunk_{len(chunks) + 1:04d}",
                **current_metadata,
                "content": part,
                "token_count": count_tokens(part)
            })

    for record in records:
        metadata = {
            key: value
            for key, value in record.items()
            if key != "content"
        }

        # Do not combine different heading groups
        group = (
            record.get("document_id"),
            record.get("section"),
            record.get("subsection"),
            record.get("sub_subsection")
        )

        previous_group = None
        if current_metadata is not None:
            previous_group = (
                current_metadata.get("document_id"),
                current_metadata.get("section"),
                current_metadata.get("subsection"),
                current_metadata.get("sub_subsection")
            )

        if current_paragraphs and group != previous_group:
            save_chunk()
            current_paragraphs = []

        current_metadata = metadata
        paragraph = record["content"]

        # Start a new chunk if the next paragraph will exceed the limit
        candidate = "\n\n".join(current_paragraphs + [paragraph])

        if current_paragraphs and count_tokens(candidate) > CHUNK_SIZE:
            save_chunk()
            current_paragraphs = []

        current_paragraphs.append(paragraph)

    # Save the final chunk
    save_chunk()

    return chunks


with open(INPUT_FILE, "r", encoding="utf-8") as file:
    cleaned_records = json.load(file)

chunks = create_chunks(cleaned_records)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(chunks, file, indent=4, ensure_ascii=False)

print("Input records:", len(cleaned_records))
print("Chunks created:", len(chunks))
print("Saved to:", OUTPUT_FILE)

print("\nFirst chunk:")
print(json.dumps(chunks[0], indent=4))

token_counts = [chunk["token_count"] for chunk in chunks]

print("\n--- Chunk Quality Statistics ---")

print("Total chunks:", len(chunks))
print("Minimum tokens:", min(token_counts))
print("Maximum tokens:", max(token_counts))
print("Average tokens:", round(sum(token_counts) / len(token_counts), 2))

print("\nChunks exceeding 400 tokens:")

for chunk in chunks:
    if chunk["token_count"] > CHUNK_SIZE:
        print(chunk["chunk_id"], chunk["token_count"])

print("\nChunks by document:")

document_counts = {}

for chunk in chunks:
    doc_id = chunk["document_id"]
    document_counts[doc_id] = document_counts.get(doc_id, 0) + 1

for doc_id, count in document_counts.items():
    print(doc_id, count)