import argparse
import json

def create_verifier_input(instructions_file: str, output_file: str) -> None:
    """Create input for verification function generation."""
    try:
        with open(instructions_file, 'r') as f:
            instructions = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: Instructions file {instructions_file} not found")
        exit(1)
    
    with open(output_file, 'w') as f:
        for instruction in instructions:
            prompt = f'''You are an expert for writing evaluation functions in Python to evaluate whether a response strictly follows an instruction.
Here is the instruction: {instruction}
Please write a Python function named `evaluate` to evaluate whether an input string `response` follows this instruction. If it follows, simply return True, otherwise return False.
Please respond with a single JSON including the evaluation function in the key `func`, 
and a list of three test cases in the key `cases`, which includes an input in the key `input` and an expected output in the key `output` (true, false).
Here is an example of output JSON format: {{\"func\": JSON_STR(use only \\\\n instead of \\n), \"cases\": [{{\"input\": str, \"output\": str}}]}}.'''

            prompt_code = open("model_prompts/create_verifiers_prompt.txt").read().strip()
            prompt_code = prompt_code.format(instruction=instruction)
            print(f"\nPROMPT:\n{prompt_code}\n")
            data = {
                'prompt': prompt_code,
                'instruction': instruction
            }
            f.write(json.dumps(data) + '\n')
    
    print(f"Created verifier input with {len(instructions)} instructions at {output_file}")

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