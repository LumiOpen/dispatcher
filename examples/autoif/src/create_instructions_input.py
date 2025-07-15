import argparse
import json

def create_input_file(seed_file: str, output_file: str, num_instructions: int = 100) -> None:
    """Create input file with a prompt for instruction augmentation."""
    try:
        with open(seed_file, 'r') as f:
            seed_instructions = f.read()
    except FileNotFoundError:
        print(f"Error: Seed file {seed_file} not found")
        exit(1)
        
    with open(output_file, 'w') as f:
        prompt = open("model_prompts/create_instructions_prompt.txt").read().strip()
        prompt = prompt.format(seed_instructions=seed_instructions.strip(), num_instructions=num_instructions)
        print(f"\nPROMPT:\n{prompt}\n")
        data = {'prompt': prompt}
        json.dump(data, f)
    
    print(f"Created input file with a prompt to generate {num_instructions} instructions in {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Create input file for instruction generation')
    
    parser.add_argument('--seed_file', type=str, required=True,
                        help='File with seed instructions')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output JSONL file with prompts for inference')
    parser.add_argument('--num_instructions', type=int, default=4,
                        help='Number of instructions to generate')
    
    args = parser.parse_args()
    
    create_input_file(args.seed_file, args.output_file, args.num_instructions)

if __name__ == "__main__":
    main()