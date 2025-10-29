"""
Evaluation script for prompt translations.
We evaluate the correctness of answers generated from translated prompts.
This acts as a proxy for the prompt quality if the answering model is fixed.

We assume that the answers are in reasoning format with the final answer after the </think> tag.
The final answer extraction and verification is done with math-verify.

The generation file should contain the generated answers in a "generated_answers" field.
The gold file should contain the reference answer in an "answer" field.

Usage: python evaluate_prompts.py <answers_path> <gold_path>"""
from math_verify import parse, verify

# Get the answers from the generated data from the file provided by the user
import json
import sys

answer_path = sys.argv[1]
gold_path = sys.argv[2]
with open(answer_path, "r") as f:
    answers = [json.loads(line) for line in f]

# Store reference answers into a dict for easy lookup
with open(gold_path, "r") as f:
    gold_data = {item["id"]: item["answer"] for item in [json.loads(line) for line in f]}

print(f"Loaded {len(answers)} answers from {answer_path}")
correct = 0
total = 0
# TODO: What metric should we actually use here?
for ans in answers:
    # There can be multiple answers in the "answers" field
    for answer in ans.get("generated_answers", []):
        # Extract the final answer from the reasoning format
        try:
            final_answer = answer.split("</think>")[-1].strip()
        except:
            print(f"Could not extract final answer from: {answer}")
            continue # TODO: Count as incorrect?
         # Qwen 3 produces [Solution section] heading with our current prompt, so we can also split on that
        final_answer = final_answer.split("[Solution section]")[-1].strip()
        gold = parse(gold_data[ans["id"]])
        pred = parse(final_answer)
        if verify(gold, pred):
            correct += 1
        else:
            print(f"Incorrect answer:\nPROMPT:\n{ans['generated_translation']}\nGOLD ANSWER:\n{gold_data[ans['id']]}\nPREDICTED ANSWER:\n{final_answer}\n")
        total += 1
    # import pdb; pdb.set_trace()
total = 400 # Count failed answers as incorrect # FIXME: Remove this hack
print(f"Accuracy: {correct / total * 100:.2f}% ({correct}/{total})")