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
            prompt = f'''You are an expert for writing instructions. Please provide 50 different instructions that meet the following requirements:
- Instructions are about the format but not style of a response
- Whether instructions are followed can be easily evaluated by a Python function

Do not generate instructions about writing style, using metaphor, or translation. Here are some examples of instructions we do not need:
- Incorporate a famous historical quote seamlessly into your answer
- Translate your answer into Pig Latin
- Use only words that are also a type of food
- Respond with a metaphor in every sentence
- Write the response as if you are a character from a Shakespearean play

Here are some examples of instructions we need:
{seed_instructions}

Please generate one instruction per line in your response.
'''
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