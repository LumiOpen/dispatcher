"""
Evaluation script for prompt translations.
We evaluate the correctness of answers generated from translated prompts.
This acts as a proxy for the prompt quality if the answering model is fixed.

We assume that the answers are in reasoning format with the final answer after the </think> tag.
The final answer extraction and verification is done with math-verify.

The generation file should contain the generated answers in a "generated_answers" field.
The gold file should contain the reference answer in an "answer" field.

Usage: python evaluate_prompts.py <answers_path> <gold_path>"""
from typing import Any


import enum
import argparse
from math_verify import parse, verify

# Get the answers from the generated data from the file provided by the user
import json
import sys

def main(answer_path, gold_path, key="generated_solution_given_traces", flag="extract_from_full_reasoning_answer"):
    with open(answer_path, "r") as f:
        answers = [json.loads(line) for line in f]

    # Store reference answers into a dict for easy lookup
    with open(gold_path, "r") as f:
        gold_data = {item["id"]: item["answer"] for item in [json.loads(line) for line in f]}

    print(f"Loaded {len(answers)} answers from {answer_path}")
    correct = 0
    total = 0
    pass_at_4 = 0
    # TODO: What metric should we actually use here?

    for samples in answers:
        sample_id = samples.get("id", None)
        # There can be multiple answers in the "answers" field
        passed = False
        final_answer = None
        for ans_id, answer in enumerate(samples.get(key, [])):
            if flag == "extract_from_full_reasoning_answer":
                # Expecting a full format containing both traces and final answer, so trying to extract just the answer part
                if "</think>" in answer:
                    final_answer = answer.split("</think>")[-1].strip()
                    total += 1
                else:
                    print(f"Could not extract final answer {ans_id} for sample {sample_id} because it does not contain closing </think> tag. Not counting towards total.")
                    continue 
            elif flag == "expect_no_additional_reasoning":
                # Extract the final answer from the reasoning format
                if "</think>" not in answer:
                    final_answer = answer
                    total += 1
                else:
                    print(f"Skipping answer {ans_id} for sample {sample_id} because it contains additional reasoning. Not counting towards total.")
                    continue 
            elif flag == "no_extraction":
                final_answer = answer
                total += 1
            else:
                print(f"Invalid flag: {flag}")
                exit(1)

            # Qwen 3 sometimes produces [Solution section] heading, so we can also split on that
            final_answer = final_answer.split("[Solution section]")[-1].strip()
            gold = parse(gold_data[sample_id])
            pred = parse(final_answer)
            if verify(gold, pred):
                correct += 1
                passed = True
        if passed:
            pass_at_4 += 1

    print(f"Accuracy: {correct / total * 100:.2f}% ({correct}/{total})")

    print(f"Pass@4: {pass_at_4 / len(answers) * 100:.2f}% ({pass_at_4}/{len(answers)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation script for prompt translations. Evaluates correctness of answers generated from translated prompts."
    )
    parser.add_argument("answer_path", type=str, help="Path to the file containing generated answers")
    parser.add_argument(
        "--gold_path", 
        type=str, 
        default="/scratch/project_462000963/posttraining_data/DeepScaleR-Preview-Dataset/default-train-sample-100.jsonl",
        help="Path to the file containing reference answers (default: /scratch/project_462000353/posttraining_data/DeepScaleR-Preview-Dataset/default-train-sample-100.jsonl)"
    )
    parser.add_argument(
        "--key",
        type=str,
        default="generated_solution_given_traces",
        help="Key to extract answers from the generated data (default: generated_solution_given_traces)"
    )
    parser.add_argument(
        "--flag",
        type=str,
        default="extract_from_full_reasoning_answer",
        choices=["expect_no_additional_reasoning", "extract_from_full_reasoning_answer", "no_extraction"],
        help="Flag to control answer extraction method (default: extract_from_full_reasoning_answer)"
    )
    
    args = parser.parse_args()
    main(args.answer_path, args.gold_path, args.key, args.flag)