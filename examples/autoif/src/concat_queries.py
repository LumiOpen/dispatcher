import argparse
from typing import List, Dict, Optional, Tuple
import random
import json
from datasets import load_dataset

def load_verifiers(verifiers_path: str) -> List[Dict]:
    """Load verifiers from a JSONL file."""
    verifiers_list = []
    try:
        with open(verifiers_path, 'r') as f:
            for line in f:
                verifiers_list.append(json.loads(line))
    except Exception as e:
        print(f"Error loading verifiers: {e}")
        return []
    return verifiers_list

def extract_query_from_messages(item: Dict) -> Optional[Tuple[str, str]]:
    """Extract the first user message and assistant response from chat messages format.
    
    Returns:
        Tuple of (user_message, assistant_response) or None if not found
    """
    if 'messages' not in item:
        return None
    
    user_message = None
    assistant_response = None
    
    for msg in item['messages']:
        if msg.get('role') == 'user' and user_message is None:
            user_message = msg.get('content', '')
        elif msg.get('role') == 'assistant' and user_message is not None and assistant_response is None:
            assistant_response = msg.get('content', '')
            break  # Stop after finding the first user-assistant pair
    
    return (user_message, assistant_response) if user_message else None

def parse_query_from_item(item: Dict, messages_format: bool, query_column_name: str, 
                         response_column_name: str, query_max_len: int) -> Optional[Dict]:
    """Parse a single query item from either standard or messages format.
    
    Returns:
        Dict with 'query', 'response', and 'metadata' keys, or None if invalid
    """
    query = {}
    
    if messages_format:
        # Extract from chat messages format
        result = extract_query_from_messages(item)
        if not result:
            return None
        
        user_message, assistant_response = result
        if not user_message or len(user_message) >= query_max_len:
            return None
        
        query['query'] = user_message
        if assistant_response:
            query['response'] = assistant_response
        # Add the rest as metadata (excluding messages)
        query['metadata'] = {k: v for k, v in item.items() if k != 'messages'}
    else:
        # Standard format
        if query_column_name not in item or len(item[query_column_name]) >= query_max_len:
            return None
        
        query['query'] = item[query_column_name]
        if response_column_name in item:
            query['response'] = item[response_column_name]
        # Add the rest as metadata
        query['metadata'] = {k: v for k, v in item.items() 
                           if k not in [query_column_name, response_column_name]}
    
    return query

def load_queries_from_file(queries_file: str, messages_format: bool, query_column_name: str,
                          response_column_name: str, query_max_len: int) -> Tuple[List[Dict], int]:
    """Load queries from a JSONL file.
    
    Returns:
        Tuple of (queries_list, skipped_count)
    """
    queries = []
    skipped_queries = 0
    
    with open(queries_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            
            try:
                item = json.loads(line)
                query = parse_query_from_item(item, messages_format, query_column_name, 
                                            response_column_name, query_max_len)
                if query:
                    queries.append(query)
                else:
                    skipped_queries += 1
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON line: {e}")
                skipped_queries += 1
                continue
    
    return queries, skipped_queries

def load_queries_from_dataset(queries_dataset: str, query_column_name: str,
                             response_column_name: str, query_max_len: int) -> Tuple[List[Dict], int]:
    """Load queries from a HuggingFace dataset.
    
    Returns:
        Tuple of (queries_list, skipped_count)
    """
    queries = []
    skipped_queries = 0
    
    dataset = load_dataset(queries_dataset)
    for item in dataset['train']:
        if query_column_name not in item or len(item[query_column_name]) >= query_max_len:
            skipped_queries += 1
            continue
        
        query = {
            'query': item[query_column_name],
            'metadata': {k: v for k, v in item.items() 
                        if k not in [query_column_name, response_column_name]}
        }
        
        if response_column_name in item:
            query['response'] = item[response_column_name]
        
        queries.append(query)
    
    return queries, skipped_queries

def select_instructions(verifiers_list: List[Dict], instructions_per_query: int,
                       instruction_usage_count: Dict[int, int]) -> List[Dict]:
    """Select instructions maintaining uniform distribution with random pairing.
    
    Returns:
        List of selected verifier dictionaries
    """
    # Find the minimum usage count
    min_usage = min(instruction_usage_count.values())
    
    # If we need more instructions than available with minimum usage,
    # also include instructions with the next lowest usage count, and so on
    selected_indices = []
    current_usage = min_usage
    
    while len(selected_indices) < instructions_per_query:
        # Get all indices with current usage count that aren't already selected
        available_indices = [i for i, count in instruction_usage_count.items() 
                           if count == current_usage and i not in selected_indices]
        
        if not available_indices:
            # Move to next usage level
            current_usage += 1
            continue
        
        # Randomly select from available indices
        needed = min(instructions_per_query - len(selected_indices), len(available_indices))
        selected_from_current = random.sample(available_indices, needed)
        selected_indices.extend(selected_from_current)
    
    # Update usage counts and collect selected verifiers
    selected_verifiers = []
    for idx in selected_indices:
        instruction_usage_count[idx] += 1
        selected_verifiers.append(verifiers_list[idx])
    
    return selected_verifiers

def create_output_entry(query: Dict, selected_verifiers: List[Dict], source: str = None) -> Dict:
    """Create the output dictionary for a single query-instruction pair."""
    # Format instructions as bullet points
    instructions_text = "\n".join([f"- {v['instruction']}" for v in selected_verifiers])
    
    prompt_template = open("model_prompts/generate_response_prompt.txt").read().strip()
    prompt = prompt_template.format(query=query['query'], instructions=instructions_text)
    
    return {
        'instruction_ids': [v['instruction_id'] for v in selected_verifiers],
        'instructions': [v['instruction'] for v in selected_verifiers],
        'query': query['query'],
        'query_response': query.get('response', ''),
        'query_metadata': query['metadata'],
        'eval_funcs': [v['eval_func'] for v in selected_verifiers],
        'cases': [v['cases'] for v in selected_verifiers],
        'prompt': prompt,
        'source': source
    }

def concat_queries(
    verifiers_path: str, 
    queries_file: str, 
    queries_dataset: str,
    query_max_len: int,
    query_column_name: str,
    response_column_name: str,
    output_file: str,
    num_of_output_lines: int = 100,
    instructions_per_query: int = 1,
    messages_format: bool = False
) -> int:
    """Concatenate queries with verification functions."""
    # Load verifiers
    verifiers_list = load_verifiers(verifiers_path)
    if not verifiers_list:
        return 0
    
    # Load queries from file or dataset
    queries = []
    skipped_queries = 0
    
    if queries_file is not None:
        queries, skipped_queries = load_queries_from_file(
            queries_file, messages_format, query_column_name, 
            response_column_name, query_max_len
        )
    elif queries_dataset is not None:
        queries, skipped_queries = load_queries_from_dataset(
            queries_dataset, query_column_name, response_column_name, query_max_len
        )
    else:
        print("No queries file or dataset provided")
        return 0
    
    print("Skipped queries due to length or missing fields:", skipped_queries)
    
    # Ensure we have some queries
    if len(queries) < 10:
        print("Warning: Very few queries available")
    
    # Track usage count for uniform distribution
    instruction_usage_count = {i: 0 for i in range(len(verifiers_list))}
    # Track instruction combinations for summary
    instruction_combination_count = {}
    
    # Generate concatenated entries
    count = 0
    query_index = 0
    
    with open(output_file, 'w') as f:
        for _ in range(num_of_output_lines):
            # Reuse queries if we've exhausted them
            query = queries[query_index % len(queries)]
            query_index += 1
            
            # Select instructions
            selected_verifiers = select_instructions(
                verifiers_list, instructions_per_query, instruction_usage_count
            )
            
            # Track instruction combination for summary
            instruction_key = tuple(sorted([v['instruction_id'] for v in selected_verifiers]))
            instruction_combination_count[instruction_key] = instruction_combination_count.get(instruction_key, 0) + 1
            
            # Create output entry
            output = create_output_entry(query, selected_verifiers, source=queries_file if queries_file else queries_dataset)
            
            f.write(json.dumps(output) + '\n')
            count += 1
    
    # Print summary statistics
    print(f"Generated {count} query-instruction pairs to {output_file}")
    print(f"Used {len(queries)} unique queries (reused {max(0, count - len(queries))} times)")
    print(f"Instruction usage distribution: {dict(instruction_usage_count)}")
    # Log instruction combination usage to file
    # with open("logs/concat_queries.log", "w") as log_file:
    #     log_file.write(f"Instruction combination usage:\n")
    #     for combination, usage_count in sorted(instruction_combination_count.items()):
    #         log_file.write(f"  {combination}: {usage_count} queries\n")
    #     print(f"Logged instruction combinations to logs/concat_queries.log")
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
    parser.add_argument('--query_max_len', type=int, default=200,
                        help='Maximum query length in characters')
    parser.add_argument('--query_column_name', type=str, default='instruction',
                        help='Column name of the desired response from the query dataset') 
    parser.add_argument('--response_column_name', type=str, default='response',
                        help='Column name of the desired response from the query dataset')                  
    parser.add_argument('--num_of_output_lines', type=int, default=100,
                        help='Number of output lines to generate (will reuse queries if needed)')
    parser.add_argument('--instructions_per_query', type=int, default=1,
                        help='Number of instructions to combine with each query (formatted as bullet points)')
    parser.add_argument('--messages_format', action='store_true',
                        help='Parse queries from chat messages format (extracts first user message)')
    
    args = parser.parse_args()

    # Create query+instruction dataset
    concat_queries(
        args.verifiers_file,
        args.queries_file,
        args.queries_dataset,
        args.query_max_len,
        args.query_column_name,
        args.response_column_name,
        args.output_file,
        args.num_of_output_lines,
        args.instructions_per_query,
        args.messages_format
    )

if __name__ == "__main__":
    main()