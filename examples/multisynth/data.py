import collections
import copy
from datasets import load_dataset
import json

orig = load_dataset("spyysalo/nemotron-cc-10K-sample")

orig_data = {}
for row in orig["train"]:
    orig_data[ row["warc_record_id"] ] = row

print(len(orig_data), " rows loaded")



# we have human judges for these languages so far.
targets = set([
    "Bulgarian",
    "Czech",
    "Finnish",
    "French",
    "German",
    "Italian",
    "Polish",
    "Spanish",
    "Swedish",
])


def create_row(model, orig_row, row):
    assert row["url"] == orig_row["url"]
    assert row["label"] == orig_row["label"]
    output_row = {
        "warc_record_id": row["warc_record_id"],
        "url": row["url"],
        "label": orig_row["label"],
        "source_language": "English",
        "target_language": row["language"],
        "translation_model": model,
        "source_text": orig_row["text"],
        "target_text": row["text"],
    }
    return output_row

def get_records(model, orig_data, model_data, languages=None):
    for row in model_data["train"]:
        if languages is not None and row["language"] not in languages:
            continue
        orig_row = orig_data[ row["warc_record_id"] ]
        assert row["url"] == orig_row["url"]
        assert row["label"] == orig_row["label"]
        output_row = create_row(model, orig_row, row)
        yield output_row



with open("merged.jsonl", "w") as f:
    models = [
        "OPUS-MT",
        "EuroLLM-9B-Instruct",
        "gemma-3-4b-it",
        "Mistral-Small-3.2-24B-Instruct-2506",
        "Qwen3-32B",
    ]
    for model in models:
        print(f"processing {model}")
        model_data = load_dataset(
            "openeurollm/nemotron-cc-10K-sample-translated",
            revision=model,
        )
        for row in get_records(model, orig_data, model_data, languages=targets):
            f.write(json.dumps(row) + "\n")

    model = "Unbabel/Tower-Plus-72B"
    model_data = load_dataset("maxidl/nemotron-cc-10k-sample-translated-tower72-26langs")
    for row in get_records(model, orig_data, model_data, languages=targets):
        f.write(json.dumps(row) + "\n")
