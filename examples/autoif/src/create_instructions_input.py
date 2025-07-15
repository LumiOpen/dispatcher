import argparse
import json

def create_input_file(seed_file: str, output_file: str, num_prompts: int = 4) -> None:
    """Create input file with prompts for instruction generation."""
    try:
        with open(seed_file, 'r') as f:
            seed_instructions = f.read()
    except FileNotFoundError:
        print(f"Error: Seed file {seed_file} not found")
        exit(1)
        
    with open(output_file, 'w') as f:
        for _ in range(num_prompts):
            prompt = open("model_prompts/create_instructions_prompt.txt").read().strip()
            prompt = prompt.format(seed_instructions=seed_instructions.strip())
            print(f"\nPROMPT:\n{prompt}\n")
            data = {'prompt': prompt}
            f.write(json.dumps(data) + '\n')
    
    print(f"Created input file with {num_prompts} prompts at {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Create input file for instruction generation')
    
    parser.add_argument('--seed_file', type=str, required=True,
                        help='File with seed instructions')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output JSONL file with prompts for inference')
    parser.add_argument('--num_prompts', type=int, default=4,
                        help='Number of prompts to generate (50 instructions each)')
    
    args = parser.parse_args()
    
    create_input_file(args.seed_file, args.output_file, args.num_prompts)

if __name__ == "__main__":
    main()