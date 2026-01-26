import argparse
import json
from src.utils.lang_id import get_language_name
import os

# Category metadata - description and subcategories
CATEGORY_METADATA = {
    "Content": {
        "description": "Instructions related to the content, grammar, and literary devices used within the text.",
        "subcategories": ["Word Properties", "Sentence Structure", "Punctuation", "Identifiers/Placeholders", "Grammatical Properties", "Keywords"],
        "additional_prompt_context": "model_prompts/additional_for_content_category.txt"
    },
    "Length": {
        "description": "Instructions related to the structure and arrangement of the output text.",
        "subcategories": ["Word Count", "Sentence Count", "Paragraph Count", "Repetition", "Uniqueness"],
        "additional_prompt_context": None
    },
    "Format": {
        "description": "Instructions related to the formatting of the response, including JSON, CSV, indentation, and special characters.",
        "subcategories": ["JSON", "CSV", "Indentation", "Special Characters", "XML", "Markdown", "Table"],
        "additional_prompt_context": None
    }
}


def create_input_file(seed_file: str, output_file: str, num_instructions_per_category: int = 100, language: str = "en") -> None:
    """Create input file with prompts for instruction augmentation from JSONL seed file."""
    
    # Load seed instructions from JSONL
    seed_instructions = []
    try:
        with open(seed_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    seed_instructions.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: Seed file {seed_file} not found")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in seed file: {e}")
        exit(1)
    
    # Group instructions by category
    instructions_by_category = {}
    for instr in seed_instructions:
        category = instr.get('category', 'Unknown')
        if category not in instructions_by_category:
            instructions_by_category[category] = []
        instructions_by_category[category].append(instr)
    
    # Load prompt template
    try:
        with open("model_prompts/create_categorised_instructions_prompt.txt", 'r', encoding='utf-8') as f:
            prompt_template = f.read().strip()
    except FileNotFoundError:
        print("Error: Prompt template not found")
        exit(1)
    
    # Create prompts for each category
    with open(output_file, 'w', encoding='utf-8') as f:
        for category, instructions in instructions_by_category.items():
            # Get category metadata
            metadata = CATEGORY_METADATA.get(category, {
                "description": f"Instructions for {category} category.",
                "subcategories": [],
                "additional_prompt_context": None
            })
            
            # Format seed instructions as JSONL examples
            seed_instructions_text = '\n'.join(json.dumps(instr, ensure_ascii=False) for instr in instructions)
            
            # Load additional context if specified
            additional_context = ""
            if metadata.get("additional_prompt_context"):
                try:
                    with open(metadata["additional_prompt_context"], 'r', encoding='utf-8') as ctx_file:
                        additional_context = ctx_file.read().strip()
                except FileNotFoundError:
                    print(f"Warning: Additional context file {metadata['additional_prompt_context']} not found")
            
            # Format the prompt
            prompt = prompt_template.format(
                language=get_language_name(language, language),
                seed_instructions=seed_instructions_text,
                num_instructions=num_instructions_per_category,
                category=category,
                description=metadata["description"],
                subcategories=", ".join(metadata["subcategories"]),
                additional_context=additional_context
            )
            
            print(f"\n{'='*60}")
            print(f"PROMPT for {category} ({len(instructions)} seed instructions):")
            print(f"{'='*60}")
            print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
            
            data = {'prompt': prompt, 'category': category}
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print(f"\nCreated input file with prompts for {len(instructions_by_category)} categories")
    print(f"Requesting {num_instructions_per_category} instructions per category")
    print(f"Output: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Create input file for instruction generation')
    
    parser.add_argument('--seed-file', type=str, required=True,
                        help='JSONL file with seed instructions (instruction, category, placeholders per line)')
    parser.add_argument('--output-file', type=str, required=True,
                        help='Output JSONL file with prompts for inference (one line per category)')
    parser.add_argument('--num-instructions-per-category', type=int, default=4,
                        help='Number of instructions to generate per category')
    parser.add_argument('--language', type=str, default="en",
                        help='Language for constructing the prompt')
    
    args = parser.parse_args()

    create_input_file(args.seed_file, args.output_file, args.num_instructions_per_category, args.language)


if __name__ == "__main__":
    main()
