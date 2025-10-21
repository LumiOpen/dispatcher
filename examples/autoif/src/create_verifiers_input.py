import argparse
import json
from jinja2 import Environment, FileSystemLoader
import os

def create_verifier_input(instructions_file: str, output_file: str) -> None:
    """Create input for verification function generation from JSONL file."""
    instructions = []
    try:
        with open(instructions_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    instruction = data.get('instruction', '').strip()
                    instruction_id = data.get('id', '')
                    category = data.get('category', '').strip()

                    if not instruction:
                        print(f"Warning: Line {line_num} has no instruction, skipping")
                        continue

                    instructions.append({
                        'id': instruction_id,
                        'instruction': instruction,
                        'category': category
                    })
                except json.JSONDecodeError as e:
                    print(f"Warning: Line {line_num} is not valid JSON, skipping: {e}")
                    continue
    except FileNotFoundError:
        print(f"Error: Instructions file {instructions_file} not found")
        exit(1)
    except Exception as e:
        print(f"Error reading JSONL file: {e}")
        exit(1)
    
    # Set up Jinja2 environment
    template_dir = 'model_prompts'
    env = Environment(loader=FileSystemLoader(template_dir))

    # Load the Jinja2 template
    try:
        template = env.get_template('create_verifiers_prompt.j2')
    except Exception as e:
        print(f"Error: Could not load Jinja2 template: {e}")
        exit(1)
    
    with open(output_file, 'w') as f:
        for item in instructions:
            # Determine if we should use the keyword branch
            has_keywords = item['category'].lower() == 'keyword'
            prompt = template.render(instruction=item['instruction'], has_keywords=has_keywords)
            
            data = {
                'instruction_id': item['id'],
                'instruction': item['instruction'],
                'instruction_category': item['category'],
                'prompt': prompt
            }
            f.write(json.dumps(data) + '\n')
    
    print(f"Created verifier input with {len(instructions)} instructions at {output_file}")
    keyword_count = sum(1 for item in instructions if item['category'].lower() == 'keyword')
    print(f"Instructions with keyword category: {keyword_count}")
    print(f"Instructions with other/no category: {len(instructions) - keyword_count}")

def main():
    parser = argparse.ArgumentParser(description='Create input for verification function generation')

    parser.add_argument('--instructions_file', type=str, required=True,
                        help='JSONL file with filtered instructions (output from augmentation post-processing)')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output JSONL file with prompts for verification function generation')

    args = parser.parse_args()

    create_verifier_input(args.instructions_file, args.output_file)

if __name__ == "__main__":
    main()