
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "output" / "chunks_v2.json"

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    chunks = json.load(file)

target_documents = [
    "POL-CARD-003",
    "POL-ACD-010",
    "POL-KYC-005"
]

for document_id in target_documents:

    print("\n" + "=" * 90)
    print("DOCUMENT:", document_id)
    print("=" * 90)

    matching_chunks = [
        chunk for chunk in chunks
        if chunk.get("document_id") == document_id
    ]

    for chunk in matching_chunks:

        print("\nChunk ID:", chunk["chunk_id"])
        print("Section:", chunk.get("section"))
        print("Subsection:", chunk.get("subsection"))
        print("Tokens:", chunk.get("token_count"))
        print("-" * 70)
        print(chunk["content"])
