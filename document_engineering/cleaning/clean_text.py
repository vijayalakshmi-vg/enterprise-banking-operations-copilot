
import re


def clean_text(text):

    # Remove unnecessary leading and trailing spaces
    text = text.strip()

    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)

    return text


def clean_records(parsed_data):

    cleaned_data = []

    for record in parsed_data:

        cleaned_record = record.copy()

        cleaned_record["content"] = clean_text(record["content"])

        cleaned_data.append(cleaned_record)

    return cleaned_data

