import argparse
from typing import List, Dict, Optional, Tuple
import random
import json
import re
import os.path
from datasets import load_dataset
from src.utils.query_skip_tracker import QuerySkipTracker


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

def extract_query_from_messages(item: Dict, turns: int = 1, messages_key: str = 'messages', no_followup: bool = False) -> Optional[Tuple[List[str], List[str]]]:
    """Extract multiple user messages and assistant responses from chat messages format.
    
    Args:
        item: Dictionary containing messages key
        turns: Number of conversation turns to extract
        messages_key: Key name for the messages list (default: 'messages')
        no_followup: If True, only first turn needs a query, rest will be padded
    
    Returns:
        Tuple of (user_messages_list, assistant_responses_list) or None if not found
    """
    if messages_key not in item:
        return None, "key_not_in_item"
    
    user_messages = []
    assistant_responses = []
    
    current_turn = 0
    expecting_user = True
    
    for msg in item[messages_key]:
        if current_turn >= turns:
            break
            
        if expecting_user and msg.get('role') == 'user':
            user_messages.append(msg.get('content', ''))
            expecting_user = False
        elif not expecting_user and msg.get('role') == 'assistant':
            assistant_responses.append(msg.get('content', ''))
            expecting_user = True
            current_turn += 1
    
    required_turns = 1 if no_followup else turns
    
    # Check if we have enough turns based on no_followup setting
    if len(user_messages) >= required_turns:
        # Pad remaining turns with empty strings if using no_followup
        while len(user_messages) < turns:
            user_messages.append('')
            
        # Pad assistant responses if needed
        while len(assistant_responses) < turns:
            assistant_responses.append('')
            
        return (user_messages[:turns], assistant_responses[:turns]), ""
    
    return None, "not_enough_turns"

def parse_query_from_item(item: Dict, messages_format: bool, query_column_name: str, 
                         response_column_name: str, query_max_len: int, turns: int = 1, 
                         messages_key: str = 'messages', no_followup: bool = False) -> Optional[Dict]:
    """Parse a single query item from either standard or messages format.
    
    Returns:
        Dict with 'queries', 'responses', and 'metadata' keys, or None if invalid
    """
    
    query = {}
    
    if messages_format:
        # Extract from chat messages format
        result, reason = extract_query_from_messages(item, turns, messages_key, no_followup)
        if not result:
            return None, reason
        
        user_messages, assistant_responses = result
        
        # Check if any user message exceeds max length (only check non-empty messages)
        if any(len(msg) >= query_max_len for msg in user_messages if msg):
            return None, "length"
        
        query['queries'] = user_messages
        query['responses'] = assistant_responses
        # Add the rest as metadata (excluding messages)
        query['metadata'] = {k: v for k, v in item.items() if k not in [messages_key, 'openai_moderation']}
    else:
        # Standard format
        if query_column_name not in item:
            return None, "key_not_in_item"
        if len(item[query_column_name]) >= query_max_len:
            return None, "length"
        
        # For non-messages format with multiple turns, we expect the data to be structured differently
        # For now, we'll treat single-turn as before, and multi-turn will need specific handling
        if turns == 1:
            query['queries'] = [item[query_column_name]]
            if response_column_name in item:
                query['responses'] = [item[response_column_name]]
            else:
                query['responses'] = ['']
        else:
            # For multi-turn non-messages format, we need the data to contain lists or structured turns
            # With no_followup, we only need the first query since subsequent turns use rephrase prompts
            required_queries = 1 if no_followup else turns
            
            if isinstance(item.get(query_column_name), list) and len(item[query_column_name]) >= required_queries:
                # Take only the required number of queries
                query['queries'] = item[query_column_name][:required_queries]
                # Pad with empty strings for turns that don't need queries (when no_followup=True)
                while len(query['queries']) < turns:
                    query['queries'].append('')
                    
                if response_column_name in item and isinstance(item[response_column_name], list):
                    responses = item[response_column_name][:turns]
                    # Pad with empty strings if needed
                    while len(responses) < turns:
                        responses.append('')
                    query['responses'] = responses
                else:
                    query['responses'] = [''] * turns
            else:
                return None, "invalid_format"
        
        # Add the rest as metadata
        query['metadata'] = {k: v for k, v in item.items() 
                           if k not in [query_column_name, response_column_name]}
    
    return query, ""

def load_queries_from_file(queries_file: str, messages_format: bool, query_column_name: str,
                          response_column_name: str, query_max_len: int, turns: int = 1,
                          messages_key: str = 'messages', no_followup: bool = False) -> Tuple[List[Dict], int]:
    """Load queries from a JSONL file.
    
    Returns:
        Tuple of (queries_list, skipped_count)
    """
    queries = []
    skip_tracker = QuerySkipTracker()
    
    with open(queries_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            
            try:
                item = json.loads(line)
                query, reason = parse_query_from_item(item, messages_format, query_column_name, 
                                            response_column_name, query_max_len, turns, messages_key, no_followup)
                if query:
                    queries.append(query)
                else:
                    skip_tracker.skip(reason)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON line: {e}")
                skip_tracker.skip('json_decode_error')
                continue

    return queries, skip_tracker

def load_queries_from_dataset_or_file(queries_dataset: str, query_column_name: str,
                                      response_column_name: str, query_max_len: int, turns: int = 1, 
                                      messages_format: bool = False, messages_key: str = 'messages',
                                      no_followup: bool = False) -> Tuple[List[Dict], int]:
    """Load queries from either a local file or a HuggingFace dataset.
    
    Determines the loading strategy based on whether the path exists locally.
    If the path exists as a local file, loads as JSONL file.
    Otherwise, attempts to load as a HuggingFace dataset.
    
    Returns:
        Tuple of (queries_list, skipped_count)
    """
    # Check if the path exists as a local file
    if os.path.exists(queries_dataset):
        print(f"Loading queries from local file: {queries_dataset}")
        return load_queries_from_file(
            queries_dataset, messages_format, query_column_name, 
            response_column_name, query_max_len, turns, messages_key, no_followup
        )
    else:
        print(f"Loading queries from HuggingFace dataset: {queries_dataset}")
        return load_queries_from_dataset(
            queries_dataset, query_column_name, response_column_name, 
            query_max_len, turns, no_followup
        )

def load_queries_from_dataset(queries_dataset: str, query_column_name: str,
                             response_column_name: str, query_max_len: int, turns: int = 1, 
                             no_followup: bool = False) -> Tuple[List[Dict], int]:
    """Load queries from a HuggingFace dataset.
    
    Returns:
        Tuple of (queries_list, skipped_count)
    """
    queries = []
    skip_tracker = QuerySkipTracker()
    
    dataset = load_dataset(queries_dataset)
    for item in dataset['train']:
        if query_column_name not in item:
            skip_tracker.skip('key_not_in_item')
            continue
        if len(item[query_column_name]) >= query_max_len:
            skip_tracker.skip('length')
            continue
        
        # For multi-turn with no_followup, we only need one query
        if turns == 1:
            query = {
                'queries': [item[query_column_name]],  # Convert to list format
                'metadata': {k: v for k, v in item.items() 
                            if k not in [query_column_name, response_column_name]}
            }
            
            if response_column_name in item:
                query['responses'] = [item[response_column_name]]
            else:
                query['responses'] = ['']
        else:
            # Multi-turn logic for datasets
            required_turns = 1 if no_followup else turns
            
            if isinstance(item.get(query_column_name), list):
                if len(item[query_column_name]) < required_turns:
                    return None, "not_enough_turns"
                # Take the required number of queries and pad if needed
                queries = item[query_column_name][:required_turns]
                while len(queries) < turns:
                    queries.append('')
                query = {
                    'queries': queries,
                    'metadata': {k: v for k, v in item.items() 
                                if k not in [query_column_name, response_column_name]}
                }
            else:
                # Single string query
                if no_followup:
                    query = {
                        'queries': [item[query_column_name]] + [''] * (turns - 1),
                        'metadata': {k: v for k, v in item.items() 
                                    if k not in [query_column_name, response_column_name]}
                    }
                else:
                    return None, "invalid_format"
            
            # Handle responses for multi-turn
            if response_column_name in item:
                if isinstance(item[response_column_name], list):
                    responses = item[response_column_name][:turns]
                    while len(responses) < turns:
                        responses.append('')
                    query['responses'] = responses
                else:
                    query['responses'] = [item[response_column_name]] + [''] * (turns - 1)
            else:
                query['responses'] = [''] * turns
        
        queries.append(query)

    return queries, skip_tracker

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

def select_instructions_multi_turn(verifiers_list: List[Dict], instructions_per_query: int,
                                  instruction_usage_count: Dict[int, int], turns: int,
                                  used_instructions_in_conversation: set = None) -> List[List[Dict]]:
    """Select instructions for multiple turns with accumulation across turns.
    
    For each turn, the instruction list includes all instructions from previous turns plus new ones.
    For example, if instructions_per_query=1 and turns=2:
    - Turn 0: [instruction_0]
    - Turn 1: [instruction_0, instruction_1]
    
    Args:
        verifiers_list: List of all available verifiers
        instructions_per_query: Number of NEW instructions to add per turn
        instruction_usage_count: Global usage count for uniform distribution
        turns: Number of turns to generate instructions for
        used_instructions_in_conversation: Set of instruction indices already used in this conversation
    
    Returns:
        List of lists - accumulated instruction lists per turn
    """
    if used_instructions_in_conversation is None:
        used_instructions_in_conversation = set()
    
    all_turn_instructions = []
    accumulated_verifiers = []
    
    for turn in range(turns):
        # Find the minimum usage count among unused instructions
        available_indices = [i for i in range(len(verifiers_list)) 
                           if i not in used_instructions_in_conversation]
        
        if len(available_indices) < instructions_per_query:
            # If we don't have enough unused instructions, we'll have to reuse some
            # Reset and use all available
            available_indices = list(range(len(verifiers_list)))
            used_instructions_in_conversation = set()
        
        min_usage = min(instruction_usage_count[i] for i in available_indices)
        
        # Select NEW instructions for this turn
        selected_indices = []
        current_usage = min_usage
        
        while len(selected_indices) < instructions_per_query:
            # Get available indices with current usage count
            candidates = [i for i in available_indices
                         if instruction_usage_count[i] == current_usage and i not in selected_indices]
            
            if not candidates:
                current_usage += 1
                continue
            
            # Randomly select from candidates
            needed = min(instructions_per_query - len(selected_indices), len(candidates))
            selected_from_current = random.sample(candidates, needed)
            selected_indices.extend(selected_from_current)
        
        # Update usage counts and tracking
        new_verifiers = []
        for idx in selected_indices:
            instruction_usage_count[idx] += 1
            used_instructions_in_conversation.add(idx)
            new_verifiers.append(verifiers_list[idx])
        
        # Accumulate instructions: add new instructions to previous ones
        accumulated_verifiers.extend(new_verifiers)
        # Create a copy of accumulated verifiers for this turn
        all_turn_instructions.append(accumulated_verifiers.copy())
    
    return all_turn_instructions

def create_output_entry(query: Dict, selected_verifiers: List[Dict] = None, source: str = None, 
                       turns: int = 1, selected_verifiers_multi_turn: List[List[Dict]] = None,
                       no_followup: bool = False) -> Dict:
    """Create the output dictionary for a single query-instruction pair or multi-turn conversation."""
    
    # For single turn, wrap selected_verifiers in a list to unify with multi-turn logic
    if turns == 1:
        if selected_verifiers is None:
            raise ValueError("selected_verifiers must be provided for single-turn conversations")
        verifiers_by_turn = [selected_verifiers]
    else:
        if selected_verifiers_multi_turn is None:
            raise ValueError("selected_verifiers_multi_turn must be provided for multi-turn conversations")
        verifiers_by_turn = selected_verifiers_multi_turn
    
    # Build prompts and collect data for each turn
    prompts = []
    all_instruction_ids = []
    all_instructions = []
    all_eval_funcs = []
    all_cases = []
    
    for turn_idx in range(turns):
        turn_verifiers = verifiers_by_turn[turn_idx]
        
        # Instructions are now already accumulated in verifiers_by_turn
        # No need for special accumulation logic here
        current_instructions_text = "\n".join([f"- {v['instruction']}" for v in turn_verifiers])
        
        # Select appropriate prompt template based on turn and no_followup flag
        if turns == 1:
            # Single turn uses the original template
            template_file = "model_prompts/generate_response_prompt.txt"
            turn_query = query['queries'][turn_idx]
            prompt_template = open(template_file).read().strip()
            prompt = prompt_template.format(query=turn_query, instructions=current_instructions_text)
        elif turn_idx == 0:
            # First turn of multi-turn conversation
            template_file = "model_prompts/generate_response_turn1_prompt.txt"
            turn_query = query['queries'][turn_idx]
            prompt_template = open(template_file).read().strip()
            prompt = prompt_template.format(query=turn_query, instructions=current_instructions_text)
        else:
            # Subsequent turns of multi-turn conversation
            if no_followup:
                # Use rephrase prompt (no query needed)
                template_file = "model_prompts/rephrase_response_turnN_prompt.txt"
                prompt_template = open(template_file).read().strip()
                prompt = prompt_template.format(instructions=current_instructions_text)
            else:
                # Use regular turnN prompt with query
                template_file = "model_prompts/generate_response_turnN_prompt.txt"
                turn_query = query['queries'][turn_idx]
                prompt_template = open(template_file).read().strip()
                prompt = prompt_template.format(query=turn_query, instructions=current_instructions_text)
        
        prompts.append(prompt)
        
        # Collect instruction data for this turn (already accumulated)
        all_instruction_ids.append([v['instruction_id'] for v in turn_verifiers])
        all_instructions.append([v['instruction'] for v in turn_verifiers])
        all_eval_funcs.append([v['eval_func'] for v in turn_verifiers])
        all_cases.append([v['cases'] for v in turn_verifiers])
    
    # Create unified output format (all keys are plural)
    return {
        'instruction_ids': all_instruction_ids,
        'instructions': all_instructions,
        'queries': query['queries'],
        'queries_responses': query['responses'],
        'query_metadata': query['metadata'],
        'eval_funcs': all_eval_funcs,
        'cases': all_cases,
        'prompts': prompts,
        'source': source
    }

def concat_queries(
    verifiers_path: str, 
    queries_dataset: str,
    query_max_len: int,
    query_column_name: str,
    response_column_name: str,
    output_file: str,
    num_output_lines: int = None,
    instructions_per_query: int = 1,
    messages_format: bool = False,
    turns: int = 1,
    messages_key: str = 'messages',
    no_followup: bool = False
) -> int:
    """Concatenate queries with verification functions."""
    # Load verifiers
    verifiers_list = load_verifiers(verifiers_path)
    if not verifiers_list:
        return 0
    
    # Load queries from file or dataset
    queries = []
    
    if queries_dataset is not None:
        queries, skip_tracker = load_queries_from_dataset_or_file(
            queries_dataset, query_column_name, response_column_name, 
            query_max_len, turns, messages_format, messages_key, no_followup
        )
    else:
        print("No queries dataset provided")
        return 0

    skip_tracker.print_summary()
    print(f"Total passed: {len(queries)}")
    
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
    
    # Determine the number of iterations
    if num_output_lines is None:
        # Process all queries once without repetition
        num_iterations = len(queries)
    else:
        # Use the specified number of output lines
        num_iterations = num_output_lines
    
    with open(output_file, 'w') as f:
        for _ in range(num_iterations):
            # Handle query selection based on mode
            if num_output_lines is None:
                # No repetition mode - use each query once
                if query_index >= len(queries):
                    break  # No more queries available
                query = queries[query_index]
                query_index += 1
            else:
                # Reuse queries if we've exhausted them
                query = queries[query_index % len(queries)]
                query_index += 1
            
            if turns == 1:
                # Single turn logic (backward compatibility)
                selected_verifiers = select_instructions(
                    verifiers_list, instructions_per_query, instruction_usage_count
                )
                
                # Track instruction combination for summary
                instruction_key = tuple(sorted([v['instruction_id'] for v in selected_verifiers]))
                instruction_combination_count[instruction_key] = instruction_combination_count.get(instruction_key, 0) + 1
                
                # Create output entry
                output = create_output_entry(query, selected_verifiers, 
                                           source=queries_dataset,
                                           turns=turns, no_followup=no_followup)
            else:
                # Multi-turn logic
                selected_verifiers_multi_turn = select_instructions_multi_turn(
                    verifiers_list, instructions_per_query, instruction_usage_count, turns
                )
                
                # Track instruction combination for summary (flatten all turns)
                all_instruction_ids = []
                for turn_verifiers in selected_verifiers_multi_turn:
                    all_instruction_ids.extend([v['instruction_id'] for v in turn_verifiers])
                instruction_key = tuple(sorted(all_instruction_ids))
                instruction_combination_count[instruction_key] = instruction_combination_count.get(instruction_key, 0) + 1
                
                # Create output entry
                output = create_output_entry(query, None, 
                                           source=queries_dataset,
                                           turns=turns,
                                           selected_verifiers_multi_turn=selected_verifiers_multi_turn,
                                           no_followup=no_followup)
            
            f.write(json.dumps(output) + '\n')
            count += 1
    
    # Print summary statistics
    print(f"Generated {count} query-instruction pairs to {output_file}")
    if num_output_lines is None:
        print(f"Used {count} unique queries (processed all available queries once)")
    else:
        print(f"Used {len(queries)} unique queries (reused {max(0, count - len(queries))} times)")
    print(f"Number of turns per conversation: {turns}")
    print(f"Instruction usage distribution: {dict(instruction_usage_count)}")
    
    return count

def main():
    parser = argparse.ArgumentParser(description='Cross-validate verifiers and concatenate with queries')
    
    parser.add_argument('--verifiers_file', type=str, required=True,
                        help='Input file with filtered verifiers')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output file for concatenated queries and verifiers')
    parser.add_argument('--queries_dataset', type=str, required=True,
                        help='Dataset with queries for concatenation. Can be a HuggingFace dataset path or a path to local jsonl file')
    parser.add_argument('--query_max_len', type=int, default=200,
                        help='Maximum query length in characters')
    parser.add_argument('--query_column_name', type=str, default='instruction',
                        help='Column name of the desired response from the query dataset') 
    parser.add_argument('--response_column_name', type=str, default='response',
                        help='Column name of the desired response from the query dataset')                  
    parser.add_argument('--num_output_lines', type=int, default=None,
                        help='Number of output lines to generate (will reuse queries if needed). If not provided, processes all queries once without repetition.')
    parser.add_argument('--instructions_per_query', type=int, default=1,
                        help='Number of instructions to combine with each query (formatted as bullet points)')
    parser.add_argument('--messages_format', action='store_true',
                        help='Parse queries from chat messages format (extracts first user message)')
    parser.add_argument('--messages_key', type=str, default='messages',
                        help='Key name for the messages list when using messages_format (default: messages)')
    parser.add_argument('--turns', type=int, default=1,
                        help='Number of conversation turns to build multi-turn prompts (default: 1)')
    parser.add_argument('--no-followup', action='store_true',
                        help='For multi-turn conversations, use rephrase prompts for turns after the first (no queries needed)')
    
    args = parser.parse_args()

    # Create query+instruction dataset
    concat_queries(
        args.verifiers_file,
        args.queries_dataset,
        args.query_max_len,
        args.query_column_name,
        args.response_column_name,
        args.output_file,
        args.num_output_lines,
        args.instructions_per_query,
        args.messages_format,
        args.turns,
        args.messages_key,
        args.no_followup
    )

if __name__ == "__main__":
    main()