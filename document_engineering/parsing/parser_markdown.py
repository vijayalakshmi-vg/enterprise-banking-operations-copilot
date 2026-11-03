
from pathlib import Path

docs = Path(r"C:\Users\14694\OneDrive\Documents\Projects\Enterprise_Banking_Operations_Copilot\document_engineering")

policy = docs / "policies" / "customer_information_policy.md"

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

# Extracting metadata
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

print("Document metadata:")
print(metadata)

print("\nTotal parsed records:", len(parsed_data))

print("\nFirst 3 parsed records:")
for item in parsed_data[:3]:
    print(item)
