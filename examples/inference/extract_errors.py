#!/usr/bin/env python3
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Extract errored samples from a JSONL file")
    parser.add_argument("input", help="Path to the input JSONL file")
    parser.add_argument("-o", "--output", default=None, help="Path to the output JSON file (default: <input>.errors.json)")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input.rsplit(".", 1)[0] + ".errors.json"

    errors = []
    with open(args.input) as f:
        for line in f:
            record = json.loads(line)
            if "__ERROR__" not in record:
                continue
            err = record["__ERROR__"]
            original = json.loads(err["original_content"])
            errors.append({
                "errored_id": original["prompt_id"],
                "error_reason": err["error"],
                "errored_messages": original["messages"],
            })

    with open(args.output, "w") as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)

    print(f"Extracted {len(errors)} errored samples to {args.output}")


if __name__ == "__main__":
    main()
