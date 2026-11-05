
import sys
import json
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from cleaning.clean_text import clean_records


# Project paths
docs = Path(
    r"C:\Users\14694\OneDrive\Documents\Projects\Enterprise_Banking_Operations_Copilot\document_engineering"
)

policy_folder = docs / "policies"
output_file = docs / "output" / "cleaned_records.json"


def process_policy(policy):

    contents = policy.read_text(encoding="utf-8")

    # Metadata labels
    metadata_labels = {
        "**Document ID:**": "document_id",
        "**Document Version:**": "document_version",
        "**Department:**": "department",
        "**Document Owner:**": "document_owner",
        "**Status:**": "status",
        "**Effective Date:**": "effective_date",
        "**Review Status:**": "review_status"
    }

    # Extract metadata
    metadata = {}

    for line in contents.splitlines():
        for label, key in metadata_labels.items():
            if line.startswith(label):
                value = line.split(label, 1)[1].strip()
                metadata[key] = value

    # Heading variables
    document_title = ""
    current_section = ""
    current_subsection = ""
    current_sub_subsection = ""

    parsed_data = []
    paragraph = []

    for line in contents.splitlines():

        # Skip metadata lines
        if any(line.startswith(label) for label in metadata_labels):
            continue

        # Identify headings
        if line.startswith("#"):

            if paragraph:
                parsed_data.append({
                    "document_title": document_title,
                    "section": current_section,
                    "subsection": current_subsection,
                    "sub_subsection": current_sub_subsection,
                    "content": " ".join(paragraph)
                })

                paragraph = []

            heading_level = len(line) - len(line.lstrip("#"))
            heading_name = line.lstrip("#").strip()

            if heading_level == 1:
                document_title = heading_name

            elif heading_level == 2:
                current_section = heading_name
                current_subsection = ""
                current_sub_subsection = ""

            elif heading_level == 3:
                current_subsection = heading_name
                current_sub_subsection = ""

            elif heading_level == 4:
                current_sub_subsection = heading_name

        elif line.strip() and line.strip() != "---":
            paragraph.append(line.strip())

        elif paragraph:
            parsed_data.append({
                "document_title": document_title,
                "section": current_section,
                "subsection": current_subsection,
                "sub_subsection": current_sub_subsection,
                "content": " ".join(paragraph)
            })

            paragraph = []

    # Save final paragraph
    if paragraph:
        parsed_data.append({
            "document_title": document_title,
            "section": current_section,
            "subsection": current_subsection,
            "sub_subsection": current_sub_subsection,
            "content": " ".join(paragraph)
        })

    # Clean parsed records
    cleaned_data = clean_records(parsed_data)

    # Attach document metadata to every record
    for record in cleaned_data:
        record.update(metadata)

    print(f"{policy.name}: {len(parsed_data)} parsed records, "
          f"{len(cleaned_data)} cleaned records")

    return cleaned_data


def main():

    all_records = []

    # Find all Markdown policy files
    policy_files = sorted(policy_folder.glob("*.md"))

    print("Total policy files found:", len(policy_files))

    # Process each policy
    for policy in policy_files:

        records = process_policy(policy)

        all_records.extend(records)

    # Save combined records
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(all_records, file, indent=4, ensure_ascii=False)

    print("\nProcessing completed!")
    print("Total policies processed:", len(policy_files))
    print("Total cleaned records:", len(all_records))
    print("Saved to:", output_file)


if __name__ == "__main__":
    main()
