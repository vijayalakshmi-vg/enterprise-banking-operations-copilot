
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from google import genai

# Project paths
ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = ROOT.parent

INPUT_FILE = ROOT / "policies" / "policy_catalog.json"
OUTPUT_FILE = ROOT / "output" / "policy_embeddings.json"

# Load API key
load_dotenv(PROJECT_ROOT / ".env")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_NAME = "gemini-embedding-001"


def generate_policy_embeddings(policies):

    embedded_policies = []

    batch_size = 10

    for start in range(0, len(policies), batch_size):

        batch = policies[start:start + batch_size]

        descriptions = [
            policy["description"] for policy in batch
        ]

        response = client.models.embed_content(
            model=MODEL_NAME,
            contents=descriptions
        )

        for policy, embedding in zip(batch, response.embeddings):

            embedded_policies.append({
                **policy,
                "embedding": embedding.values
            })

        print(
            f"Processed {len(embedded_policies)} / {len(policies)} policies"
        )

    return embedded_policies


def main():

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        policies = json.load(file)

    embedded_policies = generate_policy_embeddings(policies)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(embedded_policies, file, indent=4)

    print("\nPolicy embeddings generated successfully!")
    print(f"Total policies: {len(embedded_policies)}")
    print(f"Embedding dimensions: {len(embedded_policies[0]['embedding'])}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
