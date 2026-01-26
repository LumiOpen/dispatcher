import argparse
import json
import re
import random


def load_user_queries(queries_file: str) -> list:
    """Load user queries from a JSONL file.
    
    Supports formats:
    - {"messages": [{"role": "user", "content": "..."}]}  (lmsys format)
    - {"query": "..."} or {"question": "..."} etc.
    - Plain text lines
    """
    queries = []
    
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                
                # Handle lmsys format: messages[0].content
                if 'messages' in data and isinstance(data['messages'], list):
                    for msg in data['messages']:
                        if msg.get('role') == 'user' and 'content' in msg:
                            queries.append(msg['content'])
                            break
                # Handle simple string
                elif isinstance(data, str):
                    queries.append(data)
                # Handle dict with common keys
                elif isinstance(data, dict):
                    for key in ('query', 'question', 'prompt', 'text', 'content'):
                        if key in data:
                            queries.append(data[key])
                            break
            except json.JSONDecodeError:
                # Treat as plain text
                queries.append(line)
    
    return queries


def extract_placeholders_from_instruction(instruction: str, placeholder_metadata: dict) -> list:
    """Extract placeholder info from instruction text and metadata."""
    placeholder_names = re.findall(r'\{(\w+)\}', instruction)
    
    placeholders = []
    for name in placeholder_names:
        if name in placeholder_metadata:
            meta = placeholder_metadata[name]
            ptype = meta.get('type', 'unknown')
            
            if ptype == 'static':
                placeholders.append({'name': name, 'type': 'static (sampled from predefined values)'})
            elif ptype == 'numeric':
                min_val = meta.get('min', 1)
                max_val = meta.get('max', 10)
                placeholders.append({'name': name, 'type': f'numeric (integer between {min_val} and {max_val})'})
            elif ptype == 'dynamic':
                placeholders.append({'name': name, 'type': 'dynamic (generated based on context)'})
            else:
                placeholders.append({'name': name, 'type': 'unknown'})
        else:
            if name in ('N', 'M', 'min_count', 'max_count', 'percentage', 'min_length'):
                placeholders.append({'name': name, 'type': 'numeric (integer)'})
            elif name in ('keywords', 'forbidden_words', 'keyword', 'word', 'phrase', 'end_phrase', 'topic'):
                placeholders.append({'name': name, 'type': 'dynamic (generated based on context)'})
            else:
                placeholders.append({'name': name, 'type': 'static or dynamic (check instruction context)'})
    
    return placeholders


def create_verifier_input(instructions_file: str, queries_file: str, output_file: str, 
                          num_queries_per_instruction: int = 3) -> None:
    """Create input for verification function generation."""
    
    # Load all user queries
    all_queries = load_user_queries(queries_file)
    if not all_queries:
        print(f"Error: No user queries found in {queries_file}")
        exit(1)
    print(f"Loaded {len(all_queries)} user queries from {queries_file}")
    
    # Load instructions
    instructions = []
    with open(instructions_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                instruction = data.get('instruction', '').strip()
                instruction_id = data.get('instruction_id', str(line_num - 1))
                category = data.get('category', '').strip()
                placeholders_meta = data.get('placeholders', {})

                if not instruction:
                    print(f"Warning: Line {line_num} has no instruction, skipping")
                    continue

                instructions.append({
                    'instruction_id': instruction_id,
                    'instruction': instruction,
                    'instruction_category': category,
                    'placeholders': placeholders_meta,
                })
            except json.JSONDecodeError as e:
                print(f"Warning: Line {line_num} is not valid JSON, skipping: {e}")
                continue
    
    # Write output - sample a few queries per instruction
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in instructions:
            # Sample random queries for this instruction
            sampled_queries = random.sample(all_queries, min(num_queries_per_instruction, len(all_queries)))
            
            data = {
                'instruction_id': item['instruction_id'],
                'instruction': item['instruction'],
                'instruction_category': item['instruction_category'],
                'placeholders': item['placeholders'],
                'user_queries': sampled_queries
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    with_placeholders = sum(1 for item in instructions if item['placeholders'])
    
    print(f"Created verifier input with {len(instructions)} instructions at {output_file}")
    print(f"  With placeholders: {with_placeholders}")
    print(f"  Without placeholders: {len(instructions) - with_placeholders}")
    print(f"  User queries per instruction: {num_queries_per_instruction}")


def main():
    parser = argparse.ArgumentParser(description='Create input for verification function generation')

    parser.add_argument('--instructions-file', type=str, required=True,
                        help='JSONL file with instructions')
    parser.add_argument('--queries-file', type=str, required=True,
                        help='JSONL file with user queries (lmsys format or simple)')
    parser.add_argument('--output-file', type=str, required=True,
                        help='Output JSONL file for verifiers task')
    parser.add_argument('--num-queries', type=int, default=3,
                        help='Number of queries to sample per instruction (default: 3)')

    args = parser.parse_args()

    create_verifier_input(args.instructions_file, args.queries_file, args.output_file,
                          args.num_queries)


if __name__ == "__main__":
    main()
