#!/usr/bin/env python3
import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path


SAMPLE_SIZE = 30_000
TEST_SIZE = 150
SAMPLE_SEED = 12_345
TEST_SEED = 67_890

DATASETS = {
    "successful_score4plus_1turn_or_2turn_final_messages": (
        "successful_turn_count in {1,2}; exactly that many usable final turns; all usable scores in {4,5}"
    ),
    "highest_quality_score5_1turn_final_messages": (
        "successful_turn_count == 1; one usable final turn; usable score == 5"
    ),
    "highest_quality_score5_2turn_final_messages": (
        "successful_turn_count == 2; two usable final turns; both usable scores == 5"
    ),
    "highest_quality_score5_1turn_or_2turn_final_messages": (
        "successful_turn_count in {1,2}; exactly that many usable final turns; all usable scores == 5"
    ),
}


def usable_turns(row):
    return [
        turn
        for turn in row.get("attempted_turns") or []
        if isinstance(turn, dict) and turn.get("usable_for_final_dataset") is True
    ]


def usable_scores(row):
    return [turn.get("score") for turn in usable_turns(row)]


def matching_dataset_names(row):
    successful_turn_count = row.get("successful_turn_count")
    if successful_turn_count not in (1, 2):
        return []

    scores = usable_scores(row)
    if len(scores) != successful_turn_count:
        return []

    dataset_names = []
    all_4plus = all(score in (4, 5) for score in scores)
    all_5 = all(score == 5 for score in scores)

    if all_4plus:
        dataset_names.append("successful_score4plus_1turn_or_2turn_final_messages")

    if all_5:
        dataset_names.append("highest_quality_score5_1turn_or_2turn_final_messages")
        if successful_turn_count == 1:
            dataset_names.append("highest_quality_score5_1turn_final_messages")
        elif successful_turn_count == 2:
            dataset_names.append("highest_quality_score5_2turn_final_messages")

    return dataset_names


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
        "constraints": {
            constraint_id: source_constraints[constraint_id]
            for constraint_id in constraint_ids
            if constraint_id in source_constraints
        },
    }


def update_reservoir(reservoir, rng, line_number, count, reservoir_size):
    if len(reservoir) < reservoir_size:
        reservoir.append(line_number)
        return

    replacement_index = rng.randint(1, count)
    if replacement_index <= reservoir_size:
        reservoir[replacement_index - 1] = line_number


def select_line_numbers(input_file):
    counts = Counter()
    score_patterns = {name: Counter() for name in DATASETS}
    sample_reservoirs = {name: [] for name in DATASETS}
    full_test_reservoirs = {name: [] for name in DATASETS}
    sample_rngs = {
        name: random.Random(f"{SAMPLE_SEED}:{name}")
        for name in DATASETS
    }
    full_test_rngs = {
        name: random.Random(f"{TEST_SEED}:full:{name}")
        for name in DATASETS
    }
    source_rows = 0

    with input_file.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue

            source_rows += 1
            row = json.loads(line)
            names = matching_dataset_names(row)
            if not names:
                continue

            scores_key = ",".join(str(score) for score in sorted(usable_scores(row)))
            for name in names:
                counts[name] += 1
                score_patterns[name][scores_key] += 1
                update_reservoir(
                    sample_reservoirs[name],
                    sample_rngs[name],
                    line_number,
                    counts[name],
                    SAMPLE_SIZE,
                )
                update_reservoir(
                    full_test_reservoirs[name],
                    full_test_rngs[name],
                    line_number,
                    counts[name],
                    TEST_SIZE,
                )

    for name in DATASETS:
        count = counts[name]
        if count < SAMPLE_SIZE:
            raise ValueError(
                f"{name} has only {count} matching rows; need {SAMPLE_SIZE}"
            )

    return source_rows, counts, score_patterns, sample_reservoirs, full_test_reservoirs


def prepare_outputs(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    handles = {}
    paths = {}

    for name in DATASETS:
        handles[name] = {}
        paths[name] = {}

        for split_name, suffix in (
            ("sample_30k", "sample_30k.split"),
            ("full", "full.split"),
        ):
            dataset_dir = output_dir / f"{name}.{suffix}"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            train_path = dataset_dir / "train.jsonl"
            test_path = dataset_dir / "test.jsonl"

            # Remove the old JSON-array test file if this script is replacing a
            # previous run from before both splits were standardized on JSONL.
            old_test_json = dataset_dir / "test.json"
            if old_test_json.exists():
                old_test_json.unlink()

            train_tmp = dataset_dir / "train.jsonl.tmp"
            test_tmp = dataset_dir / "test.jsonl.tmp"
            handles[name][split_name] = {
                "train": train_tmp.open("w", encoding="utf-8"),
                "test": test_tmp.open("w", encoding="utf-8"),
            }
            paths[name][split_name] = {
                "dir": dataset_dir,
                "train": train_path,
                "test": test_path,
                "train_tmp": train_tmp,
                "test_tmp": test_tmp,
            }

    return handles, paths


def write_splits(input_file, output_dir, sample_reservoirs, full_test_reservoirs):
    selected_sample_lines = {
        name: set(line_numbers)
        for name, line_numbers in sample_reservoirs.items()
    }
    sample_test_lines = {
        name: set(
            random.Random(f"{TEST_SEED}:{name}").sample(line_numbers, TEST_SIZE)
        )
        for name, line_numbers in sample_reservoirs.items()
    }
    full_test_lines = {
        name: set(line_numbers)
        for name, line_numbers in full_test_reservoirs.items()
    }
    train_counts = {
        "sample_30k": Counter(),
        "full": Counter(),
    }
    test_counts = {
        "sample_30k": Counter(),
        "full": Counter(),
    }

    handles, paths = prepare_outputs(output_dir)
    try:
        with input_file.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue

                row = json.loads(line)
                matching_names = matching_dataset_names(row)
                if not matching_names:
                    continue

                record_line = json.dumps(
                    compact_record(row),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ) + "\n"

                for name in matching_names:
                    if line_number in selected_sample_lines[name]:
                        if line_number in sample_test_lines[name]:
                            handles[name]["sample_30k"]["test"].write(record_line)
                            test_counts["sample_30k"][name] += 1
                        else:
                            handles[name]["sample_30k"]["train"].write(record_line)
                            train_counts["sample_30k"][name] += 1

                    if line_number in full_test_lines[name]:
                        handles[name]["full"]["test"].write(record_line)
                        test_counts["full"][name] += 1
                    else:
                        handles[name]["full"]["train"].write(record_line)
                        train_counts["full"][name] += 1
    finally:
        for dataset_handles in handles.values():
            for split_handles in dataset_handles.values():
                split_handles["train"].close()
                split_handles["test"].close()

    for dataset_paths in paths.values():
        for split_paths in dataset_paths.values():
            os.replace(split_paths["train_tmp"], split_paths["train"])
            os.replace(split_paths["test_tmp"], split_paths["test"])

    return train_counts, test_counts, paths


def write_manifest(
    output_dir,
    input_file,
    source_rows,
    counts,
    score_patterns,
    train_counts,
    test_counts,
    paths,
):
    manifest = {
        "source": str(input_file),
        "source_rows": source_rows,
        "sample_size": SAMPLE_SIZE,
        "test_size": TEST_SIZE,
        "train_size": SAMPLE_SIZE - TEST_SIZE,
        "sample_seed": SAMPLE_SEED,
        "test_seed": TEST_SEED,
        "format": "jsonl",
        "record_shape": (
            "Each row contains only messages and constraints. messages is the "
            "source final messages list; constraints is pruned to IDs from "
            "usable attempted_turns.constraint_ids."
        ),
        "filters": DATASETS,
        "splits": {
            "sample_30k": {},
            "full": {},
        },
    }

    for name in DATASETS:
        manifest["splits"]["sample_30k"][name] = {
            "matching_rows": counts[name],
            "score_patterns": dict(score_patterns[name]),
            "output_dir": str(paths[name]["sample_30k"]["dir"]),
            "train": str(paths[name]["sample_30k"]["train"]),
            "test": str(paths[name]["sample_30k"]["test"]),
            "train_rows": train_counts["sample_30k"][name],
            "test_rows": test_counts["sample_30k"][name],
        }
        manifest["splits"]["full"][name] = {
            "matching_rows": counts[name],
            "score_patterns": dict(score_patterns[name]),
            "output_dir": str(paths[name]["full"]["dir"]),
            "train": str(paths[name]["full"]["train"]),
            "test": str(paths[name]["full"]["test"]),
            "train_rows": train_counts["full"][name],
            "test_rows": test_counts["full"][name],
        }

    manifest_path = output_dir / "sample_30k_train_test_splits_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, ensure_ascii=False, indent=2, sort_keys=True)
        manifest_file.write("\n")

    return manifest_path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build compact SFT train/test splits directly from scored_responses.jsonl."
        )
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Input scored_responses.jsonl file.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where split subdirectories and the manifest will be written.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    (
        source_rows,
        counts,
        score_patterns,
        sample_reservoirs,
        full_test_reservoirs,
    ) = select_line_numbers(args.input_file)
    train_counts, test_counts, paths = write_splits(
        args.input_file,
        args.output_dir,
        sample_reservoirs,
        full_test_reservoirs,
    )
    manifest_path = write_manifest(
        args.output_dir,
        args.input_file,
        source_rows,
        counts,
        score_patterns,
        train_counts,
        test_counts,
        paths,
    )

    print(f"Wrote manifest: {manifest_path}")
    for name in DATASETS:
        print(
            f"{name}: matched={counts[name]} "
            f"sample_train={train_counts['sample_30k'][name]} "
            f"sample_test={test_counts['sample_30k'][name]} "
            f"sample_output_dir={paths[name]['sample_30k']['dir']} "
            f"full_train={train_counts['full'][name]} "
            f"full_test={test_counts['full'][name]} "
            f"full_output_dir={paths[name]['full']['dir']}"
        )


if __name__ == "__main__":
    main()
