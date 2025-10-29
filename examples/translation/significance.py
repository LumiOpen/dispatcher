"""
Statistical significance testing script for comparing two sets of prompt translations.
We evaluate the correctness of answers generated from two different translation sets
and test whether the difference in performance is statistically significant.

Usage: python evaluate_prompts_significance.py <answers1_path> <answers2_path> <gold_path>
NOTE: This is vibe coded!
"""
from math_verify import parse, verify
import json
import sys
import numpy as np
from scipy import stats
from collections import defaultdict

def extract_final_answer(answer):
    """Extract final answer from reasoning format."""
    try:
        final_answer = answer.split("</think>")[-1].strip()
        # # Qwen 3 produces [Solution section] heading with our current prompt, so we can also split on that
        final_answer = final_answer.split("[Solution section]")[-1].strip()
        return final_answer
    except:
        return None

def evaluate_translation_set(answer_path, gold_data):
    """Evaluate a single set of translations and return per-item results."""
    with open(answer_path, "r") as f:
        answers = [json.loads(line) for line in f]
    
    results = {}  # item_id__gen_id -> correctness (0/1)
    
    for ans in answers:
        item_id = ans["id"]
        for gen_id, answer in enumerate(ans.get("generated_answers", [None, None, None, None])):
            final_answer = extract_final_answer(answer)
            if final_answer is None:
                results[f"{item_id}__{gen_id}"] = 0  # Count extraction failures as incorrect
                continue
                
            gold = parse(gold_data[item_id])
            pred = parse(final_answer)
            is_correct = 1 if verify(gold, pred) else 0
            results[f"{item_id}__{gen_id}"] = is_correct
    
    return results, answers

def paired_t_test(scores1, scores2):
    """Perform paired t-test on item-level scores."""
    # Ensure we have the same items in both sets
    common_items = set(scores1.keys()) & set(scores2.keys())
    
    values1 = [scores1[item_id] for item_id in sorted(common_items)]
    values2 = [scores2[item_id] for item_id in sorted(common_items)]
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(values1, values2)
    
    return t_stat, p_value, len(common_items)

def main():
    if len(sys.argv) != 4:
        print("Usage: python evaluate_prompts_significance.py <answers1_path> <answers2_path> <gold_path>")
        sys.exit(1)
    
    answer_path1 = sys.argv[1]
    answer_path2 = sys.argv[2]
    gold_path = sys.argv[3]
    
    # Load gold data
    with open(gold_path, "r") as f:
        gold_data = {item["id"]: item["answer"] for item in [json.loads(line) for line in f]}
    
    print("Evaluating Translation Set 1...")
    scores1, answers1 = evaluate_translation_set(answer_path1, gold_data)
    accuracy1 = np.mean(list(scores1.values()))

    print("Evaluating Translation Set 2...")
    scores2, answers2 = evaluate_translation_set(answer_path2, gold_data)
    accuracy2 = np.mean(list(scores2.values()))

    print(f"\n=== RESULTS ===")
    print(f"Translation Set 1 Accuracy: {accuracy1 * 100:.2f}%")
    print(f"Translation Set 2 Accuracy: {accuracy2 * 100:.2f}%")
    print(f"Difference: {(accuracy1 - accuracy2) * 100:.2f} percentage points")
    
    # Statistical significance testing
    print(f"\n=== STATISTICAL SIGNIFICANCE TESTS ===")
    
    # Paired t-test
    t_stat, p_value_t, n_items = paired_t_test(scores1, scores2)
    print(f"Paired t-test:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value_t:.4f}")
    print(f"  n_items: {n_items}")
    print(f"  Significant at α=0.05: {'Yes' if p_value_t < 0.05 else 'No'}")

if __name__ == "__main__":
    main()