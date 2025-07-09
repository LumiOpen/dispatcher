import argparse
from typing import List, Dict
import random
import json
from datasets import load_dataset

def concat_queries(
    verifiers_path: str, 
    queries_file: str, 
    queries_dataset: str,
    output_file: str,
    queries_per_instruction: int = 1,
    verifiers_per_query: int = 1
) -> int:
    """Concatenate queries with verification functions."""
    # Load verifiers from file
    verifiers_list = []
    try:
        with open(verifiers_path, 'r') as f:
            for line in f:
                verifiers_list.append(json.loads(line))
    except Exception as e:
        print(f"Error loading verifiers: {e}")
        return 0
    
    # Load queries
    queries = []
    if queries_file is not None:
        with open(queries_file, 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) < 200:  # Filter out long queries
                    queries.append(line)
    elif queries_dataset is not None:
        dataset = load_dataset(queries_dataset)
        for item in dataset['train']:
            if 'instruction' in item and len(item['instruction']) < 200:
                queries.append(item['instruction'])
    else:
        print("No queries file or dataset provided")
        return 0
    
    # Ensure we have some queries
    if len(queries) < 10:
        print("Warning: Very few queries available")
    
    # Concatenate
    count = 0
    with open(output_file, 'w') as f:
        for verifier in verifiers_list:
            instruction = verifier['instruction']
            
            # Select queries for this instruction
            selected_queries = random.sample(queries, min(queries_per_instruction, len(queries)))
            
            for query in selected_queries:
                prompt = f"""Please answer the query strictly following the instruction.
[instruction] {instruction}
[Query] {query}"""
                
                output = {
                    'instruction': instruction,
                    'query': query,
                    'eval_func': verifier['eval_func'],
                    'cases': verifier['cases'],
                    'prompt': prompt
                }
                
                f.write(json.dumps(output) + '\n')
                count += 1
    
    print(f"Generated {count} query-instruction pairs")
    return count

def main():
    parser = argparse.ArgumentParser(description='Cross-validate verifiers and concatenate with queries')
    
    parser.add_argument('--verifiers_file', type=str, required=True,
                        help='Input file with filtered verifiers')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output file for concatenated queries and verifiers')
    parser.add_argument('--queries_file', type=str, default=None,
                        help='File with queries for concatenation')
    parser.add_argument('--queries_dataset', type=str, default=None,
                        help='Dataset with queries for concatenation')
    parser.add_argument('--queries_per_instruction', type=int, default=1,
                        help='Number of queries to use per instruction')
    parser.add_argument('--verifiers_per_query', type=int, default=1,
                        help='Maximum number of verification functions per query')
    
    args = parser.parse_args()

    # TODO support verifiers_per_query
    # Concatenate with queries
    concat_queries(
        args.verifiers_file,
        args.queries_file,
        args.queries_dataset,
        args.output_file,
        args.queries_per_instruction,
        args.verifiers_per_query
    )

if __name__ == "__main__":
    main()