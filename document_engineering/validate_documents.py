from pathlib import Path

docs = Path(r"C:\Users\14694\OneDrive\Documents\Projects\Enterprise_Banking_Operations_Copilot\document_engineering")

policy_files = docs.glob("policies/*.md")

list_policies = list(policy_files)

print(len(list_policies))

data = []

required_fields = [
    "Document ID:",
    "Document Version:",
    "Department:",
    "Document Owner:",
    "Status:",
    "Effective Date:",
    "Review Status:"
]

metadata_labels = {
    "**Document ID:**": "document_id",
    "**Document Version:**": "document_version",
    "**Department:**": "department",
    "**Document Owner:**": "document_owner",
    "**Status:**": "status",
    "**Effective Date:**": "effective_date",
    "**Review Status:**": "review_status"
}

all_metadata = []

for policies in list_policies:

    contents = policies.read_text(encoding="utf-8")

    print(policies.name)

    data.append(contents)

    metadata = {}

    missing_fields = []

    rule_ids = []

    # Checking required fields
    for field in required_fields:
        if field not in contents:
            missing_fields.append(field)

    print("Missing fields:", missing_fields)

    # Extracting metadata
    for line in contents.splitlines():

        for label, key in metadata_labels.items():

            if label in line:
                value = line.split(label, 1)[1].strip()
                metadata[key] = value

        # Extracting Rule IDs
        if "**Rule ID:**" in line:
            rule_id = line.split("**Rule ID:**", 1)[1].strip()
            rule_ids.append(rule_id)
        # Checking Rule IDs
        if len(rule_ids) == 0:
            print("No Rule IDs found")
        else:
            print("Number of Rule IDs:", len(rule_ids))
    # Checking empty metadata values
    metadata_missing_fields = []

    for key in metadata_labels.values():
        if key not in metadata or metadata[key].strip() == "":
            metadata_missing_fields.append(key)
    print("Empty metadata fields:", metadata_missing_fields)

    metadata["rule_ids"] = rule_ids
    metadata["file_name"] = policies.name
    metadata["missing_fields"] = missing_fields

    all_metadata.append(metadata)


# Printing extracted metadata
print("\nAll metadata:")
print(all_metadata)


# Checking Document ID duplicates
document_id = []

for item in all_metadata:
    document_id.append(item["document_id"])

if len(document_id) == len(set(document_id)):
    print("All Document IDs are unique")
else:
    print("Duplicate Document IDs found")


# Checking Rule ID duplicates
all_rule_ids = []

for item in all_metadata:
    all_rule_ids.extend(item["rule_ids"])

if len(all_rule_ids) == len(set(all_rule_ids)):
    print("All Rule IDs are unique")
else:
    print("Duplicate Rule IDs found")