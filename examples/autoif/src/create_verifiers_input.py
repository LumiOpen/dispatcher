import argparse
import json
import csv
from jinja2 import Environment, FileSystemLoader
import os

def create_verifier_input(instructions_file: str, output_file: str) -> None:
    """Create input for verification function generation."""
    instructions = []
    try:
        with open(instructions_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                instruction = row['instruction'].strip()
                instruction_id = row['id']
                category = row.get('category', '').strip() if 'category' in row else ''
                if instruction:
                    instructions.append({
                        'id': instruction_id, 
                        'instruction': instruction,
                        'category': category
                    })
    except FileNotFoundError:
        print(f"Error: Instructions file {instructions_file} not found")
        exit(1)
    except KeyError as e:
        print(f"Error: Required column {e} not found in CSV file")
        exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
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
                        help='File with filtered instructions')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output file with prompts for verification function generation')
    
    args = parser.parse_args()
    
    create_verifier_input(args.instructions_file, args.output_file)

if __name__ == "__main__":
    main()