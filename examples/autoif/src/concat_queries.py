import argparse
from typing import List, Dict, Optional, Tuple
import random
import json
import re
import os.path
from datasets import load_dataset
from src.utils.query_skip_tracker import QuerySkipTracker
from src.utils.lang_id import get_language_name

class InstructionSelector:
    """Class to handle instruction selection logic with uniform distribution."""
    
    def __init__(self, verifiers_list: List[Dict], balance_categories: bool = False):
        """Initialize the instruction selector.
        
        Args:
            verifiers_list: List of verifier dictionaries with 'instruction_id' and optionally 'instruction_category'
            balance_categories: Whether to balance instruction selection across categories
        """
        self.balance_categories = balance_categories
        
        # Create efficient lookup dictionary for verifiers by instruction_id
        self.verifiers = {verifier['instruction_id']: verifier for verifier in verifiers_list}
        
        # Initialize usage count using instruction_id as key
        self.instruction_usage_count = {verifier['instruction_id']: 0 for verifier in verifiers_list}
        
        # If balancing categories, create category mappings
        if balance_categories:
            self.categories = {}
            for verifier in verifiers_list:
                category = verifier.get('instruction_category', 'default')
                if category not in self.categories:
                    self.categories[category] = []
                self.categories[category].append(verifier['instruction_id'])
            self.category_usage_count = {cat: 0 for cat in self.categories.keys()}
    
    def _select_instructions_uniform(self, instructions_per_query: int) -> List[Dict]:
        """Select instructions maintaining uniform distribution across all instructions."""
        # Find the minimum usage count
        min_usage = min(self.instruction_usage_count.values())
        
        # If we need more instructions than available with minimum usage,
        # also include instructions with the next lowest usage count, and so on
        selected_ids = []
        current_usage = min_usage
        
        while len(selected_ids) < instructions_per_query:
            # Get all IDs with current usage count that aren't already selected
            available_ids = [instruction_id for instruction_id, count in self.instruction_usage_count.items() 
                           if count == current_usage and instruction_id not in selected_ids]
            
            if not available_ids:
                # Move to next usage level
                current_usage += 1
                continue
            
            # Randomly select from available IDs
            needed = min(instructions_per_query - len(selected_ids), len(available_ids))
            selected_from_current = random.sample(available_ids, needed)
            selected_ids.extend(selected_from_current)
        
        # Update usage counts and collect selected verifiers
        selected_verifiers = []
        for instruction_id in selected_ids:
            self.instruction_usage_count[instruction_id] += 1
            selected_verifiers.append(self.verifiers[instruction_id])
        
        return selected_verifiers
    
    def _select_instructions_category_balanced(self, instructions_per_query: int) -> List[Dict]:
        """Select instructions uniformly across categories and within categories."""
        selected_verifiers = []
        
        for i in range(instructions_per_query):
            # Find the category with minimum usage count to ensure uniform distribution across categories
            min_category_usage = min(self.category_usage_count.values())
            categories_with_min_usage = [cat for cat, count in self.category_usage_count.items() 
                                       if count == min_category_usage]
            
            # Randomly select one category from those with minimum usage
            selected_category = random.choice(categories_with_min_usage)
            
            # Within the selected category, find instructions with minimum usage
            category_instruction_ids = self.categories[selected_category]
            min_usage_in_category = min(self.instruction_usage_count[inst_id] for inst_id in category_instruction_ids)
            
            # Get available instructions in this category with minimum usage
            available_in_category = [inst_id for inst_id in category_instruction_ids 
                                   if self.instruction_usage_count[inst_id] == min_usage_in_category]
            
            # Randomly select one instruction from available options
            selected_id = random.choice(available_in_category)
            
            # Update usage count and add to results
            self.instruction_usage_count[selected_id] += 1
            self.category_usage_count[selected_category] += 1
            selected_verifiers.append(self.verifiers[selected_id])
        
        return selected_verifiers
    
    def select_instructions(self, instructions_per_query: int) -> List[Dict]:
        """Select instructions using the configured strategy."""
        if self.balance_categories:
            return self._select_instructions_category_balanced(instructions_per_query)
        else:
            return self._select_instructions_uniform(instructions_per_query)
    
    def select_instructions_multi_turn(self, instructions_per_query: int, turns: int,
                                     used_instructions_in_conversation: set = None) -> List[List[Dict]]:
        """Select instructions for multiple turns with accumulation across turns."""
        if used_instructions_in_conversation is None:
            used_instructions_in_conversation = set()
        
        all_turn_instructions = []
        accumulated_verifiers = []
        
        for turn in range(turns):
            n_instructions = instructions_per_query
            if instructions_per_query == 3:
                choices = [1, 2, 3]
                weights = [0.5, 0.25, 0.25]
                n_instructions = random.choices(choices, weights=weights, k=1)[0]
            # Find available instruction IDs (not used in this conversation)
            available_ids = [instruction_id for instruction_id in self.verifiers.keys() 
                           if instruction_id not in used_instructions_in_conversation]
            
            if len(available_ids) < n_instructions:
                # If we don't have enough unused instructions, reset and use all available
                available_ids = list(self.verifiers.keys())
                used_instructions_in_conversation = set()
            
            # Select NEW instructions for this turn
            if self.balance_categories:
                # For category balanced selection with multi-turn, we need to modify the approach
                # Create a temporary category mapping with only available instructions
                temp_categories = {}
                for inst_id in available_ids:
                    verifier = self.verifiers[inst_id]
                    category = verifier.get('instruction_category', 'default')
                    if category not in temp_categories:
                        temp_categories[category] = []
                    temp_categories[category].append(inst_id)
                
                selected_ids = []
                
                for i in range(n_instructions):
                    # Find the category with minimum usage count to ensure uniform distribution across categories
                    available_categories = [cat for cat in temp_categories.keys() 
                                          if any(inst_id not in selected_ids for inst_id in temp_categories[cat])]
                    
                    if not available_categories:
                        # If all categories are exhausted, pick from any available instructions
                        available_in_category = [inst_id for inst_id in available_ids 
                                               if inst_id not in selected_ids]
                        if available_in_category:
                            min_usage = min(self.instruction_usage_count[inst_id] for inst_id in available_in_category)
                            candidates = [inst_id for inst_id in available_in_category 
                                        if self.instruction_usage_count[inst_id] == min_usage]
                            selected_id = random.choice(candidates)
                            selected_ids.append(selected_id)
                        continue
                    
                    # Among available categories, find those with minimum usage
                    min_category_usage = min(self.category_usage_count[cat] for cat in available_categories)
                    categories_with_min_usage = [cat for cat in available_categories 
                                               if self.category_usage_count[cat] == min_category_usage]
                    
                    # Randomly select one category from those with minimum usage
                    selected_category = random.choice(categories_with_min_usage)
                    
                    # Within the selected category, find instructions with minimum usage
                    category_instruction_ids = temp_categories[selected_category]
                    available_in_category = [inst_id for inst_id in category_instruction_ids 
                                           if inst_id not in selected_ids]
                    
                    if available_in_category:
                        # Among available, select the one with minimum usage
                        min_usage = min(self.instruction_usage_count[inst_id] for inst_id in available_in_category)
                        candidates = [inst_id for inst_id in available_in_category 
                                    if self.instruction_usage_count[inst_id] == min_usage]
                        selected_id = random.choice(candidates)
                        selected_ids.append(selected_id)
            else:
                # Regular uniform selection
                min_usage = min(self.instruction_usage_count[inst_id] for inst_id in available_ids)
                
                selected_ids = []
                current_usage = min_usage
                
                while len(selected_ids) < n_instructions:
                    # Get available IDs with current usage count
                    candidates = [inst_id for inst_id in available_ids
                                if self.instruction_usage_count[inst_id] == current_usage and inst_id not in selected_ids]
                    
                    if not candidates:
                        current_usage += 1
                        continue
                    
                    # Randomly select from candidates
                    needed = min(n_instructions - len(selected_ids), len(candidates))
                    selected_from_current = random.sample(candidates, needed)
                    selected_ids.extend(selected_from_current)
            
            # Update usage counts and tracking
            new_verifiers = []
            for inst_id in selected_ids:
                self.instruction_usage_count[inst_id] += 1
                used_instructions_in_conversation.add(inst_id)
                new_verifiers.append(self.verifiers[inst_id])
            
            # Accumulate instructions: add new instructions to previous ones
            accumulated_verifiers.extend(new_verifiers)
            # Create a copy of accumulated verifiers for this turn
            all_turn_instructions.append(accumulated_verifiers.copy())
        
        return all_turn_instructions
    
    def get_usage_stats(self) -> Dict:
        """Get current usage statistics."""
        stats = {
            'instruction_usage': dict(self.instruction_usage_count)
        }
        if self.balance_categories:
            stats['category_usage'] = dict(self.category_usage_count)
            stats['categories'] = {cat: len(inst_ids) for cat, inst_ids in self.categories.items()}
        return stats


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
            
        return {
            "user_messages": user_messages[:turns],
            "assistant_responses": assistant_responses[:turns]
        }, ""

def parse_query_from_item(item: Dict, messages_format: bool, query_column_name: str,
                         response_column_name: str, query_max_len: int, turns: int = 1,
                         messages_key: str = 'messages', no_followup: bool = False, language: str = 'eng') -> Optional[Dict]:
    """Parse a single query item from either standard or messages format.

    Returns:
        Dict with 'queries', 'responses', and 'metadata' keys, or None if invalid
    """

    # Filter by language if language is set and 'lang' or 'language' key exists in item
    if 'lang' in item and item['lang'] != language:
        return None, f"language_mismatch_{item['lang']}_expected_{language}"
    if 'language' in item and item['language'] != language:
        return None, f"language_mismatch_{item['language']}_expected_{language}"

    query = {}
    
    if messages_format:
        # Extract from chat messages format
        result, reason = extract_query_from_messages(
            item=item, 
            turns=turns, 
            messages_key=messages_key, 
            no_followup=no_followup
        )
        if not result:
            return None, reason

        user_messages = result['user_messages']
        assistant_responses = result['assistant_responses']

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
                          messages_key: str = 'messages', no_followup: bool = False, language: str = 'eng') -> Tuple[List[Dict], int]:
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
                query, reason = parse_query_from_item(
                    item=item,
                    messages_format=messages_format,
                    query_column_name=query_column_name,
                    response_column_name=response_column_name,
                    query_max_len=query_max_len,
                    turns=turns,
                    messages_key=messages_key,
                    no_followup=no_followup,
                    language=language
                )
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
                                      no_followup: bool = False, language: str = 'eng') -> Tuple[List[Dict], int]:
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
            queries_file=queries_dataset,
            messages_format=messages_format,
            query_column_name=query_column_name,
            response_column_name=response_column_name,
            query_max_len=query_max_len,
            turns=turns,
            messages_key=messages_key,
            no_followup=no_followup,
            language=language
        )
    else:
        print(f"Loading queries from HuggingFace dataset: {queries_dataset}")
        return load_queries_from_dataset(
            queries_dataset=queries_dataset,
            messages_format=messages_format,
            query_column_name=query_column_name,
            response_column_name=response_column_name,
            query_max_len=query_max_len,
            turns=turns,
            messages_key=messages_key,
            no_followup=no_followup,
            language=language
        )

def load_queries_from_dataset(queries_dataset: str, messages_format: bool, query_column_name: str,
                             response_column_name: str, query_max_len: int, turns: int = 1,
                             messages_key: str = 'messages', no_followup: bool = False, language: str = 'eng') -> Tuple[List[Dict], int]:
    """Load queries from a HuggingFace dataset.

    Returns:
        Tuple of (queries_list, skipped_count)
    """
    queries = []
    skip_tracker = QuerySkipTracker()

    dataset = load_dataset(queries_dataset)
    for item in dataset['train']:
        query, reason = parse_query_from_item(
            item=item,
            messages_format=messages_format,
            query_column_name=query_column_name,
            response_column_name=response_column_name,
            query_max_len=query_max_len,
            turns=turns,
            messages_key=messages_key,
            no_followup=no_followup,
            language=language
        )
        if query:
            queries.append(query)
        else:
            skip_tracker.skip(reason)

    return queries, skip_tracker

def create_output_entry(query: Dict, selected_verifiers: List[Dict] = None, source: str = None, 
                       turns: int = 1, selected_verifiers_multi_turn: List[List[Dict]] = None,
                       no_followup: bool = False, language: str = "") -> Dict:
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
    all_instruction_categories = []
    all_eval_funcs = []
    
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
            prompt = prompt_template.format(query=turn_query, instructions=current_instructions_text, language=language)
        elif turn_idx == 0:
            # First turn of multi-turn conversation
            template_file = "model_prompts/generate_response_turn1_prompt.txt"
            turn_query = query['queries'][turn_idx]
            prompt_template = open(template_file).read().strip()
            prompt = prompt_template.format(query=turn_query, instructions=current_instructions_text, language=language)
        else:
            # Subsequent turns of multi-turn conversation
            if no_followup:
                # Use rephrase prompt (no query needed)
                template_file = "model_prompts/rephrase_response_turnN_prompt.txt"
                prompt_template = open(template_file).read().strip()
                prompt = prompt_template.format(instructions=current_instructions_text, language=language)
            else:
                # Use regular turnN prompt with query
                template_file = "model_prompts/generate_response_turnN_prompt.txt"
                turn_query = query['queries'][turn_idx]
                prompt_template = open(template_file).read().strip()
                prompt = prompt_template.format(query=turn_query, instructions=current_instructions_text, language=language)
        
        prompts.append(prompt)
        
        # Collect instruction data for this turn (already accumulated)
        all_instruction_ids.append([v['instruction_id'] for v in turn_verifiers])
        all_instructions.append([v['instruction'] for v in turn_verifiers])
        all_instruction_categories.append([v.get('instruction_category', 'default') for v in turn_verifiers])
        all_eval_funcs.append([v['eval_func'] for v in turn_verifiers])
    
    # Create unified output format (all keys are plural)
    return {
        'instruction_ids': all_instruction_ids,
        'instructions': all_instructions,
        'instruction_categories': all_instruction_categories,
        'queries': query['queries'],
        'queries_responses': query['responses'],
        'query_metadata': query['metadata'],
        'eval_funcs': all_eval_funcs,
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
    language: str = 'eng',
    num_output_lines: int = None,
    instructions_per_query: int = 1,
    messages_format: bool = False,
    turns: int = 1,
    messages_key: str = 'messages',
    no_followup: bool = False,
    balance_categories: bool = False
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
            queries_dataset=queries_dataset,
            query_column_name=query_column_name,
            response_column_name=response_column_name,
            query_max_len=query_max_len,
            turns=turns,
            messages_format=messages_format,
            messages_key=messages_key,
            no_followup=no_followup,
            language=language
        )
    else:
        print("No queries dataset provided")
        return 0

    skip_tracker.print_summary()
    print(f"Total passed: {len(queries)}")

    # Ensure we have some queries
    if len(queries) < 10:
        print("Warning: Very few queries available")

    # Initialize instruction selector
    instruction_selector = InstructionSelector(
        verifiers_list=verifiers_list,
        balance_categories=balance_categories
    )

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

    language_full_name = get_language_name(language, language)  # Map language code to full name
    
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
                selected_verifiers = instruction_selector.select_instructions(instructions_per_query)
                
                # Track instruction combination for summary
                instruction_key = tuple(sorted([v['instruction_id'] for v in selected_verifiers]))
                instruction_combination_count[instruction_key] = instruction_combination_count.get(instruction_key, 0) + 1
                
                # Create output entry
                output = create_output_entry(
                    query=query,
                    selected_verifiers=selected_verifiers,
                    source=queries_dataset,
                    turns=turns,
                    no_followup=no_followup,
                    language=language_full_name
                )
            else:
                # Multi-turn logic
                selected_verifiers_multi_turn = instruction_selector.select_instructions_multi_turn(
                    instructions_per_query, turns
                )
                
                # Track instruction combination for summary (flatten all turns)
                all_instruction_ids = []
                for turn_verifiers in selected_verifiers_multi_turn:
                    all_instruction_ids.extend([v['instruction_id'] for v in turn_verifiers])
                instruction_key = tuple(sorted(all_instruction_ids))
                instruction_combination_count[instruction_key] = instruction_combination_count.get(instruction_key, 0) + 1
                
                # Create output entry
                output = create_output_entry(
                    query=query,
                    selected_verifiers=None,
                    source=queries_dataset,
                    turns=turns,
                    selected_verifiers_multi_turn=selected_verifiers_multi_turn,
                    no_followup=no_followup,
                    language=language_full_name
                )
            
            f.write(json.dumps(output) + '\n')
            count += 1
    
    # Print summary statistics
    print(f"Generated {count} query-instruction pairs to {output_file}")
    if num_output_lines is None:
        print(f"Used {count} unique queries (processed all available queries once)")
    else:
        print(f"Used {len(queries)} unique queries (reused {max(0, count - len(queries))} times)")
    print(f"Number of turns per conversation: {turns}")
    
    # Print usage statistics
    usage_stats = instruction_selector.get_usage_stats()
    print(f"Instruction usage distribution: {usage_stats['instruction_usage']}")
    
    if balance_categories:
        print(f"Category usage distribution: {usage_stats['category_usage']}")
        print(f"Instructions per category: {usage_stats['categories']}")
    
    return count

def main():
    parser = argparse.ArgumentParser(description='Cross-validate verifiers and concatenate with queries')

    parser.add_argument('--verifiers-file', type=str, required=True,
                        help='Input file with filtered verifiers')
    parser.add_argument('--output-file', type=str, required=True,
                        help='Output file for concatenated queries and verifiers')
    parser.add_argument('--queries-dataset', type=str, required=True,
                        help='Dataset with queries for concatenation. Can be a HuggingFace dataset path or a path to local jsonl file')
    parser.add_argument('--language', type=str, default='eng',
                        help='Language code (default: eng)')
    parser.add_argument('--query-max-len', type=int, default=200,
                        help='Maximum query length in characters')
    parser.add_argument('--query-column-name', type=str, default='instruction',
                        help='Column name of the desired response from the query dataset')
    parser.add_argument('--response-column-name', type=str, default='response',
                        help='Column name of the desired response from the query dataset')
    parser.add_argument('--num-output-lines', type=int, default=None,
                        help='Number of output lines to generate (will reuse queries if needed). If not provided, processes all queries once without repetition.')
    parser.add_argument('--instructions-per-query', type=int, default=1,
                        help='Number of instructions to combine with each query (formatted as bullet points)')
    parser.add_argument('--messages-format', action='store_true',
                        help='Parse queries from chat messages format (extracts first user message)')
    parser.add_argument('--messages-key', type=str, default='messages',
                        help='Key name for the messages list when using messages_format (default: messages)')
    parser.add_argument('--turns', type=int, default=1,
                        help='Number of conversation turns to build multi-turn prompts (default: 1)')
    parser.add_argument('--no-followup', action='store_true',
                        help='For multi-turn conversations, use rephrase prompts for turns after the first (no queries needed)')
    parser.add_argument('--balance-categories', action='store_true',
                        help='Balance instruction selection across categories for uniform distribution within and across categories')

    args = parser.parse_args()

    # Create query+instruction dataset
    concat_queries(
        verifiers_path=args.verifiers_file,
        queries_dataset=args.queries_dataset,
        query_max_len=args.query_max_len,
        query_column_name=args.query_column_name,
        response_column_name=args.response_column_name,
        output_file=args.output_file,
        language=args.language,
        num_output_lines=args.num_output_lines,
        instructions_per_query=args.instructions_per_query,
        messages_format=args.messages_format,
        turns=args.turns,
        messages_key=args.messages_key,
        no_followup=args.no_followup,
        balance_categories=args.balance_categories
    )

if __name__ == "__main__":
    main()