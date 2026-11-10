
import json
from pathlib import Path
import tiktoken

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "output" / "cleaned_records.json"
OUTPUT_FILE = ROOT / "output" / "chunks_v2.json"

CHUNK_SIZE = 200
CHUNK_OVERLAP = 30

tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text):
    return len(tokenizer.encode(text))


def split_text(text):
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

    current_group = None
    section_parts = []
    section_metadata = None

    def save_section():
        if not section_parts:
            return

        section_text = "\n\n".join(section_parts)

        for part in split_text(section_text):
            chunks.append({
                "chunk_id": f"chunk_{len(chunks) + 1:04d}",
                **section_metadata,
                "content": part,
                "token_count": count_tokens(part)
            })

    for record in records:

        group = (
            record.get("document_id"),
            record.get("section")
        )

        if current_group is not None and group != current_group:
            save_section()
            section_parts = []

        if group != current_group:
            current_group = group

            section_metadata = {
                key: value
                for key, value in record.items()
                if key not in (
                    "content",
                    "subsection",
                    "sub_subsection"
                )
            }

        text = record["content"]

        if record.get("subsection"):
            text = (
                f"Subsection: {record['subsection']}\n"
                + text
            )

        if record.get("sub_subsection"):
            text = (
                f"Sub-subsection: {record['sub_subsection']}\n"
                + text
            )

        section_parts.append(text)

    save_section()

    return chunks


def main():

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        records = json.load(file)

    chunks = create_chunks(records)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            chunks,
            file,
            indent=4,
            ensure_ascii=False
        )

    token_counts = [
        chunk["token_count"]
        for chunk in chunks
    ]

    print("Input records:", len(records))
    print("Chunks created:", len(chunks))
    print("Minimum tokens:", min(token_counts))
    print("Maximum tokens:", max(token_counts))
    print(
        "Average tokens:",
        round(sum(token_counts) / len(token_counts), 2)
    )
    print("Saved to:", OUTPUT_FILE)

    print("\nFirst chunk:")
    print(json.dumps(chunks[0], indent=4))


if __name__ == "__main__":
    main()
