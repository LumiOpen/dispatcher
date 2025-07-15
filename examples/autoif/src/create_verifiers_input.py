import argparse
import json
import csv

def create_verifier_input(instructions_file: str, output_file: str) -> None:
    """Create input for verification function generation."""
    instructions = []
    try:
        with open(instructions_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                instruction = row['instruction'].strip()
                instruction_id = row['id']
                if instruction:
                    instructions.append({'id': instruction_id, 'instruction': instruction})
    except FileNotFoundError:
        print(f"Error: Instructions file {instructions_file} not found")
        exit(1)
    except KeyError as e:
        print(f"Error: Required column {e} not found in CSV file")
        exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        exit(1)
    
    with open(output_file, 'w') as f:
        for item in instructions:
            prompt = open("model_prompts/create_verifiers_prompt.txt").read().strip()
            prompt = prompt.format(instruction=instruction)
            print(f"\nPROMPT:\n{prompt}\n")
            data = {
                'instruction_id': item['id'],
                'instruction': item['instruction'],
                'prompt': prompt
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