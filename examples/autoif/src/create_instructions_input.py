import argparse
import json
from src.utils.lang_id import get_language_name
import os

def create_input_file(seed_file: str, output_file: str, num_instructions_per_category: int = 100, language: str = "en") -> None:
    """Create input file with a prompt for instruction augmentation."""
    try:
        with open(seed_file, 'r') as f:
            content = f.read().strip()
    except FileNotFoundError:
        print(f"Error: Seed file {seed_file} not found")
        exit(1)
    
    # Try to parse as JSON first
    try:
        seed_data = json.loads(content)
        if isinstance(seed_data, dict):
            # Check for new format with "seed_instructions" key
            if "seed_instructions" in seed_data:
                # Handle new JSON format with seed_instructions array
                with open(output_file, 'w') as f:
                    for category_data in seed_data["seed_instructions"]:
                        category = category_data["category"]
                        description = category_data["description"]
                        subcategories = ", ".join(category_data["subcategories"])
                        instructions = category_data["instructions"]
                        
                        # Convert list of instructions to newline-separated string
                        seed_instructions = '\n'.join(instructions)
                        prompt = open("model_prompts/create_categorised_instructions_prompt.txt").read().strip()

                        # Optional formatting step for additional_context
                        additional_context = ""
                        if "additional_prompt_context" in category_data and category_data["additional_prompt_context"]:
                            context_value = category_data["additional_prompt_context"]
                            # Check if the context is a path to a txt file
                            if isinstance(context_value, str) and context_value.endswith('.txt'):
                                try:
                                    with open(context_value, 'r') as context_file:
                                        additional_context = context_file.read().strip()
                                except FileNotFoundError:
                                    print(f"Warning: Context file {context_value} not found, using as literal text")
                                    additional_context = context_value
                            else:
                                additional_context = context_value
                        
                        prompt = prompt.format(
                            language=get_language_name(language, language),
                            seed_instructions=seed_instructions, 
                            num_instructions=num_instructions_per_category, 
                            category=category,
                            description=description,
                            subcategories=subcategories,
                            additional_context=additional_context
                        )
                        print(f"\nPROMPT for {category}:\n{prompt}\n")
                        data = {'prompt': prompt, 'category': category}
                        f.write(json.dumps(data) + '\n')

                print(f"Created input file with prompts for {len(seed_data['seed_instructions'])} categories to generate {num_instructions_per_category} instructions each in {output_file}")
                return
            print("seed_instructions not part of input file. Exiting.")
            exit(1)
    except json.JSONDecodeError:
        pass
    
    # Handle plain text format (original behavior)
    lines = content.strip().split('\n')
    lines = [line.strip() for line in lines if line.strip()]
    
    # Generate prompt for regular instructions
    regular_content = '\n'.join(lines)
    with open(output_file, 'w') as f:
        prompt = open("model_prompts/create_instructions_prompt.txt").read().strip()
        prompt = prompt.format(
            language=get_language_name.get(language, language),
            seed_instructions=regular_content, 
            num_instructions=num_instructions_per_category)
        print(f"\nPROMPT:\n{prompt}\n")
        data = {'prompt': prompt}
        json.dump(data, f)
    
    print(f"Created input file with a prompt to generate {num_instructions_per_category} instructions in {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Create input file for instruction generation')
    
    parser.add_argument('--seed-file', type=str, required=True,
                        help='File with seed instructions (plain text or JSON with categorized instructions)')
    parser.add_argument('--output-file', type=str, required=True,
                        help='Output JSONL file with prompts for inference (one line per category if JSON input)')
    parser.add_argument('--num-instructions-per-category', type=int, default=4,
                        help='Number of instructions to generate per category')
    parser.add_argument('--language', type=str, default="en",
                        help='For constructing the prompt')
    
    args = parser.parse_args()

    create_input_file(args.seed_file, args.output_file, args.num_instructions_per_category, args.language)

if __name__ == "__main__":
    main()