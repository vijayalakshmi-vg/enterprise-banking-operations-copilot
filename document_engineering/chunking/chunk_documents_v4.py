
import json
import re
from pathlib import Path

import tiktoken

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "output" / "cleaned_records.json"
OUTPUT_FILE = ROOT / "output" / "chunks_v4.json"

MAX_CHUNK_SIZE = 200

tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text):
    return len(tokenizer.encode(text))


def split_sentences(text):
    return [
        sentence.strip()
        for sentence in re.split(r'(?<=[.!?])\s+', text.strip())
        if sentence.strip()
    ]


def extract_rule_ids(text):
    return sorted(set(
        re.findall(r'Rule ID:\s*([A-Z0-9-]+)', text, re.IGNORECASE)
    ))


def make_id(value):
    value = re.sub(r'[^a-zA-Z0-9]+', '-', str(value))
    return value.strip('-').lower()


def create_chunks(records):

    # Group records by their smallest available topic boundary.
    groups = {}

    for record in records:

        document_id = record.get("document_id")
        section = record.get("section") or "General"

        subsection = record.get("subsection")
        sub_subsection = record.get("sub_subsection")

        topic = sub_subsection or subsection or section

        key = (
            document_id,
            section,
            topic
        )

        groups.setdefault(key, []).append(record)

    chunks = []

    for (document_id, section, topic), group in groups.items():

        parent_id = (
            f"{make_id(document_id)}-"
            f"{make_id(section)}"
        )

        # Preserve the source metadata.
        metadata = {
            "document_id": document_id,
            "section": section,
            "subsection": group[0].get("subsection"),
            "sub_subsection": group[0].get("sub_subsection"),
            "topic": topic,
            "parent_id": parent_id
        }

        # Preserve original paragraph boundaries.
        paragraphs = [
            record["content"].strip()
            for record in group
            if record.get("content", "").strip()
        ]

        sentences = []

        for paragraph in paragraphs:
            sentences.extend(split_sentences(paragraph))

        current_sentences = []

        def save_chunk():

            if not current_sentences:
                return

            raw_content = " ".join(current_sentences)

            # Add heading context to help semantic retrieval.
            contextual_content = (
                f"Policy: {document_id}\n"
                f"Section: {section}\n"
                f"Topic: {topic}\n"
                f"Content: {raw_content}"
            )

            chunks.append({
                "chunk_id": f"chunk_{len(chunks) + 1:04d}",
                **metadata,
                "rule_ids": extract_rule_ids(raw_content),
                "content": contextual_content,
                "token_count": count_tokens(contextual_content)
            })

            current_sentences.clear()

        for sentence in sentences:

            if count_tokens(sentence) > MAX_CHUNK_SIZE:

                save_chunk()

                # Fallback for unusually long sentences.
                tokens = tokenizer.encode(sentence)

                for start in range(0, len(tokens), MAX_CHUNK_SIZE):

                    part = tokenizer.decode(
                        tokens[start:start + MAX_CHUNK_SIZE]
                    )

                    current_sentences.append(part)
                    save_chunk()

                continue

            candidate = " ".join(
                current_sentences + [sentence]
            )

            if (
                count_tokens(candidate) > MAX_CHUNK_SIZE
                and current_sentences
            ):
                save_chunk()

            current_sentences.append(sentence)

        save_chunk()

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

    token_counts = [chunk["token_count"] for chunk in chunks]

    print("Input records:", len(records))
    print("V4 chunks created:", len(chunks))
    print("Minimum tokens:", min(token_counts))
    print("Maximum tokens:", max(token_counts))
    print(
        "Average tokens:",
        round(sum(token_counts) / len(token_counts), 2)
    )
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()

