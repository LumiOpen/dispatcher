import argparse
import json
import re
from collections import defaultdict
from src.utils.lang_id import detect_language


def process_output(input_file: str, output_file: str, placeholder_lookup_file: str, 
                   language: str = 'en', max_instructions: int = 100, seed_file: str = None) -> None:
    """
    Process augmented instructions output:
    1. Parse JSON output from LLM
    2. De-duplicate instructions and filter by language
    3. Build placeholder lookup table from all static placeholders
    4. Include seed instructions in the output
    """
    
    seen_instructions = set()
    instruction_count = 0
    all_instructions = []
    
    # Aggregate placeholder values across all instructions
    # Structure: {placeholder_name: {"type": "static", "values": set()}}
    placeholder_lookup = defaultdict(lambda: {"type": None, "values": set()})
    
    # Load seed instructions if provided (they serve as both examples and actual data)
    if seed_file:
        try:
            with open(seed_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        seed_instr = json.loads(line)
                        all_instructions.append(seed_instr)
                        seen_instructions.add(seed_instr['instruction'])
                        # Add placeholders to lookup
                        _update_placeholder_lookup(placeholder_lookup, seed_instr.get('placeholders', {}))
            print(f"Loaded {len(all_instructions)} seed instructions from {seed_file}")
        except FileNotFoundError:
            print(f"Warning: Seed file {seed_file} not found")
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON in seed file: {e}")
    
    # Process LLM output
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            results = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Input file contains invalid JSON: {e}")
        exit(1)
    
    # Process each category's results
    for result in results:
        category = result.get('original', {}).get('category', 'Unknown')
        
        for response in result.get('responses', []):
            # Parse JSON objects from the response
            parsed_instructions = _parse_json_instructions(response)
            
            print(f"Parsed {len(parsed_instructions)} instructions from response for category {category}")
            
            for instr_data in parsed_instructions:
                instruction = instr_data.get('instruction', '').strip()
                placeholders = instr_data.get('placeholders', {})
                
                if not instruction or len(instruction) < 10:
                    print(f"Skipping: instruction too short: {instruction[:50]}")
                    continue
                
                if instruction in seen_instructions:
                    print(f"Skipping: duplicate instruction")
                    continue
                
                # Check language
                try:
                    lang_code1, lang_code2 = detect_language(instruction)
                    if lang_code1 != language and (lang_code2 is None or lang_code2 != language):
                        print(f"Skipping: wrong language {lang_code1}/{lang_code2}, expected {language}")
                        continue
                except Exception as e:
                    print(f"Language detection error: {e}")
                    continue
                
                seen_instructions.add(instruction)
                all_instructions.append({
                    'instruction': instruction,
                    'category': category,
                    'placeholders': placeholders
                })
                
                # Update placeholder lookup
                _update_placeholder_lookup(placeholder_lookup, placeholders)
    
    # Limit total instructions
    if len(all_instructions) > int(max_instructions):
        # Distribute evenly across categories
        by_category = defaultdict(list)
        for instr in all_instructions:
            by_category[instr['category']].append(instr)
        
        per_category = int(max_instructions) // len(by_category)
        all_instructions = []
        for cat, instrs in by_category.items():
            all_instructions.extend(instrs[:per_category])
        print(f"Limited to {len(all_instructions)} instructions ({per_category} per category)")
    
    # Write output JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, instr in enumerate(all_instructions):
            instr['instruction_id'] = str(idx)
            f.write(json.dumps(instr, ensure_ascii=False) + '\n')
    
    print(f"\nOutput {len(all_instructions)} instructions to {output_file}")
    
    # Write placeholder lookup table
    lookup_output = {}
    for name, data in placeholder_lookup.items():
        if data['type'] == 'static':
            lookup_output[name] = {
                'type': 'static',
                'values': sorted(list(data['values']))
            }
        elif data['type'] == 'numeric':
            lookup_output[name] = {'type': 'numeric'}
        elif data['type'] == 'dynamic':
            lookup_output[name] = {'type': 'dynamic'}
    
    with open(placeholder_lookup_file, 'w', encoding='utf-8') as f:
        json.dump(lookup_output, f, ensure_ascii=False, indent=2)
    
    print(f"Created placeholder lookup table with {len(lookup_output)} placeholders at {placeholder_lookup_file}")
    for name, data in lookup_output.items():
        if data['type'] == 'static':
            print(f"  {name}: static ({len(data['values'])} values)")
        elif data['type'] == 'numeric':
            print(f"  {name}: numeric")
        else:
            print(f"  {name}: dynamic")


def _parse_json_instructions(response: str) -> list:
    """Parse JSON instruction objects from LLM response."""
    instructions = []
    
    # Try to find JSON objects line by line
    for line in response.split('\n'):
        line = line.strip()
        if not line or not line.startswith('{'):
            continue
        
        try:
            data = json.loads(line)
            if 'instruction' in data:
                instructions.append(data)
        except json.JSONDecodeError:
            # Try to extract JSON from the line
            match = re.search(r'\{.*\}', line)
            if match:
                try:
                    data = json.loads(match.group())
                    if 'instruction' in data:
                        instructions.append(data)
                except json.JSONDecodeError:
                    pass
    
    # If no line-by-line matches, try to find all JSON objects in the response
    if not instructions:
        for match in re.finditer(r'\{[^{}]*"instruction"[^{}]*\}', response):
            try:
                data = json.loads(match.group())
                instructions.append(data)
            except json.JSONDecodeError:
                pass
    
    return instructions


def _update_placeholder_lookup(lookup: dict, placeholders: dict) -> None:
    """Update the placeholder lookup table with values from an instruction."""
    for name, data in placeholders.items():
        if not isinstance(data, dict):
            continue
        
        ptype = data.get('type')
        
        if ptype == 'static':
            lookup[name]['type'] = 'static'
            values = data.get('values', [])
            if isinstance(values, list):
                lookup[name]['values'].update(values)
        
        elif ptype == 'numeric':
            lookup[name]['type'] = 'numeric'
        
        elif ptype == 'dynamic':
            lookup[name]['type'] = 'dynamic'


def main():
    parser = argparse.ArgumentParser(description='Process and filter generated instructions')

    parser.add_argument('--input-file', type=str, required=True,
                        help='Input file with model-generated instructions (JSONL)')
    parser.add_argument('--output-file', type=str, required=True,
                        help='Output file for filtered instructions (JSONL)')
    parser.add_argument('--placeholder-lookup-file', type=str, required=True,
                        help='Output JSON file for placeholder lookup table')
    parser.add_argument('--language', type=str, default='en',
                        help='Language code for filtering (e.g., "en" for English)')
    parser.add_argument('--max-instructions', type=int, default=100,
                        help='Maximum number of instructions to output')
    parser.add_argument('--seed-file', type=str, default=None,
                        help='Seed instructions JSONL file to include in output')

    args = parser.parse_args()

    process_output(
        input_file=args.input_file,
        output_file=args.output_file,
        placeholder_lookup_file=args.placeholder_lookup_file,
        language=args.language,
        max_instructions=args.max_instructions,
        seed_file=args.seed_file
    )


if __name__ == "__main__":
    main()
