#!/usr/bin/env python3
import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_SAMPLE_SIZE = 30_000
DEFAULT_TEST_SIZE = 150
TEST_SEED = 67_890


def usable_turns(row):
    return [
        turn
        for turn in row.get("attempted_turns") or []
        if isinstance(turn, dict) and turn.get("usable_for_final_dataset") is True
    ]


def usable_scores(row):
    return [turn.get("score") for turn in usable_turns(row)]


def is_eligible(row):
    successful_turn_count = row.get("successful_turn_count")
    return (
        isinstance(successful_turn_count, int)
        and successful_turn_count > 0
        and len(usable_turns(row)) == successful_turn_count
    )


def compact_record(row):
    constraint_ids = []
    seen = set()
    for turn in usable_turns(row):
        for constraint_id in turn.get("constraint_ids") or []:
            constraint_id = str(constraint_id)
            if constraint_id not in seen:
                seen.add(constraint_id)
                constraint_ids.append(constraint_id)

    source_constraints = row.get("constraints") or {}
    return {
        "messages": row.get("messages") or [],
        "constraints": [
            {
                "constraint_id": constraint_id,
                "constraint": source_constraints[constraint_id]["constraint"],
                "template": source_constraints[constraint_id]["template"],
                "kwargs": source_constraints[constraint_id]["kwargs"],
                "eval_funcs": source_constraints[constraint_id]["eval_funcs"],
                "category": source_constraints[constraint_id]["category"],
            }
            for constraint_id in constraint_ids
            if constraint_id in source_constraints
        ],
    }


def select_line_numbers(input_file, sample_size, test_size):
    target_size = sample_size + test_size
    source_rows = 0
    eligible_counts = Counter()
    score_patterns = Counter()
    line_numbers_by_turn_count = defaultdict(list)

    with input_file.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue

            source_rows += 1
            row = json.loads(line)
            if not is_eligible(row):
                continue

            successful_turn_count = row["successful_turn_count"]
            eligible_counts[successful_turn_count] += 1
            score_patterns[",".join(str(score) for score in usable_scores(row))] += 1

            if len(line_numbers_by_turn_count[successful_turn_count]) < target_size:
                line_numbers_by_turn_count[successful_turn_count].append(line_number)

    selected_line_numbers = []
    selected_by_turn_count = Counter()
    for turn_count in sorted(line_numbers_by_turn_count, reverse=True):
        remaining = target_size - len(selected_line_numbers)
        if remaining <= 0:
            break

        selected = line_numbers_by_turn_count[turn_count][:remaining]
        selected_line_numbers.extend(selected)
        selected_by_turn_count[turn_count] = len(selected)

    if len(selected_line_numbers) < target_size:
        raise ValueError(
            f"Only selected {len(selected_line_numbers)} eligible rows; "
            f"need {target_size}. Eligible counts: {dict(eligible_counts)}"
        )

    test_line_numbers = set(random.Random(TEST_SEED).sample(selected_line_numbers, test_size))
    train_line_numbers = set(selected_line_numbers) - test_line_numbers

    return {
        "source_rows": source_rows,
        "eligible_counts": eligible_counts,
        "score_patterns": score_patterns,
        "selected_line_numbers": set(selected_line_numbers),
        "train_line_numbers": train_line_numbers,
        "test_line_numbers": test_line_numbers,
        "selected_by_turn_count": selected_by_turn_count,
    }


def write_splits(input_file, output_dir, selected):
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    test_path = output_dir / "test.jsonl"
    train_tmp = output_dir / "train.jsonl.tmp"
    test_tmp = output_dir / "test.jsonl.tmp"

    train_counts = Counter()
    test_counts = Counter()
    train_score_counts = Counter()
    test_score_counts = Counter()

    with (
        train_tmp.open("w", encoding="utf-8") as train_file,
        test_tmp.open("w", encoding="utf-8") as test_file,
        input_file.open(encoding="utf-8") as source,
    ):
        for line_number, line in enumerate(source, 1):
            if line_number not in selected["selected_line_numbers"]:
                continue

            row = json.loads(line)
            record_line = (
                json.dumps(
                    compact_record(row),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            successful_turn_count = row["successful_turn_count"]
            scores = usable_scores(row)

            if line_number in selected["test_line_numbers"]:
                test_file.write(record_line)
                test_counts[successful_turn_count] += 1
                test_score_counts.update(scores)
            else:
                train_file.write(record_line)
                train_counts[successful_turn_count] += 1
                train_score_counts.update(scores)

    os.replace(train_tmp, train_path)
    os.replace(test_tmp, test_path)

    return {
        "train_path": train_path,
        "test_path": test_path,
        "train_counts": train_counts,
        "test_counts": test_counts,
        "train_score_counts": train_score_counts,
        "test_score_counts": test_score_counts,
    }


def write_manifest(output_dir, input_file, sample_size, test_size, selected, written):
    manifest = {
        "source": str(input_file),
        "source_rows": selected["source_rows"],
        "sample_size": sample_size,
        "test_size": test_size,
        "total_size": sample_size + test_size,
        "test_seed": TEST_SEED,
        "format": "jsonl",
        "selection": (
            "Eligible rows are sorted by descending successful_turn_count. "
            "Within each successful_turn_count bucket, source order is preserved. "
            "The test split is sampled randomly from the selected rows."
        ),
        "eligibility": (
            "successful_turn_count is a positive integer and exactly that many "
            "attempted_turns have usable_for_final_dataset == true."
        ),
        "record_shape": (
            "Each row contains messages and constraints. constraints is a list "
            "of constraint_id, constraint, template, kwargs, eval_funcs, and category."
        ),
        "eligible_counts": dict(sorted(selected["eligible_counts"].items())),
        "selected_by_successful_turn_count": dict(
            sorted(selected["selected_by_turn_count"].items())
        ),
        "source_score_patterns": dict(selected["score_patterns"]),
        "splits": {
            "train": {
                "path": str(written["train_path"]),
                "rows": sum(written["train_counts"].values()),
                "by_successful_turn_count": dict(sorted(written["train_counts"].items())),
                "score_counts": dict(sorted(written["train_score_counts"].items())),
            },
            "test": {
                "path": str(written["test_path"]),
                "rows": sum(written["test_counts"].values()),
                "by_successful_turn_count": dict(sorted(written["test_counts"].items())),
                "score_counts": dict(sorted(written["test_score_counts"].items())),
            },
        },
    }

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, ensure_ascii=False, indent=2, sort_keys=True)
        manifest_file.write("\n")

    return manifest_path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build compact SFT train/test splits from scored_responses.jsonl, "
            "prioritizing longer successful chats."
        )
    )
    parser.add_argument(
        "--input-file",
        required=True,
        type=Path,
        help="Input scored_responses.jsonl file.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where train.jsonl, test.jsonl, and manifest.json will be written.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of training samples to write. Default: {DEFAULT_SAMPLE_SIZE}.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=DEFAULT_TEST_SIZE,
        help=f"Number of random test samples to write. Default: {DEFAULT_TEST_SIZE}.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.train_size < 0:
        raise ValueError("--train-size must be non-negative")
    if args.test_size < 0:
        raise ValueError("--test-size must be non-negative")

    selected = select_line_numbers(args.input_file, args.train_size, args.test_size)
    written = write_splits(args.input_file, args.output_dir, selected)
    manifest_path = write_manifest(
        args.output_dir,
        args.input_file,
        args.train_size,
        args.test_size,
        selected,
        written,
    )

    print(f"Wrote manifest: {manifest_path}")
    print(
        f"train_rows={sum(written['train_counts'].values())} "
        f"test_rows={sum(written['test_counts'].values())} "
        f"output_dir={args.output_dir}"
    )
    print(
        "selected_by_successful_turn_count="
        f"{dict(sorted(selected['selected_by_turn_count'].items()))}"
    )


if __name__ == "__main__":
    main()
