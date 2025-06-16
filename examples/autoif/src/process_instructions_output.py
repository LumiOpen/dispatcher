import argparse
import json
import re
# from huggingface_hub import InferenceClient

def process_output(input_file: str, output_file: str, seed_file: str, language: str = 'fi') -> None:
    """Process generated instructions, filter by language, and save to file.
    
    First loads seed instructions, then appends new non-duplicate instructions in target language.
    """
    # Initialize language identifier model
    # glot_client = InferenceClient('cis-lmu/glotlid')
    
    # 1. Start with seed instructions - assume they are already clean
    clean_instructions = set()
    try:
        with open(seed_file, 'r') as f:
            for line in f:
                instruction = line.strip()
                if instruction:
                    clean_instructions.add(instruction)
        print(f"Loaded {len(clean_instructions)} seed instructions")
    except Exception as e:
        print(f"Error loading seed instructions: {e}")
        exit(1)
    
    # Write seed instructions to output file
    with open(output_file, 'w') as f:
        for instruction in clean_instructions:
            f.write(instruction + '\n')
    
    # 2. Process new instructions from model output
    try:
        with open(input_file, 'r') as f:
            results = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Input file {input_file} contains invalid JSON")
        exit(1)
    
    # Append new instructions to output file
    with open(output_file, 'a') as out_f:
        for result in results:
            for response in result.get('responses', []):
                # Extract instructions
                instructions = re.findall(r'^\d+\.?\s+(.*)', response, re.MULTILINE)
                
                # If no numbered instructions found, try line-by-line
                if not instructions:
                    instructions = [line.strip() for line in response.split('\n') 
                                  if line.strip() and not line.strip().startswith('#')]
                
                for instruction in instructions:
                    instruction = instruction.strip()
                    # Skip empty or very short instructions
                    if len(instruction) < 5:
                        continue
                    
                    # Skip duplicates
                    if instruction in clean_instructions:
                        continue
                    
                    # # Check language
                    # try:
                    #     lang = glot_client.post(json={'inputs': instruction})
                    #     if lang['detected_language'] == language:
                    #         # Add to set to track duplicates
                    #         clean_instructions.add(instruction)
                    #         # Write to file directly
                    #         out_f.write(instruction + '\n')
                    # except Exception as e:
                    #     print(f'Language detection error: {e}')

                    # Add to set to track duplicates
                    clean_instructions.add(instruction)
                    # Write to file directly
                    out_f.write(instruction + '\n')
    
    print(f'Output {len(clean_instructions)} {language} instructions')

def main():
    parser = argparse.ArgumentParser(description='Process and filter generated instructions')
    
    parser.add_argument('--input_file', type=str, required=True,
                        help='Input file with model-generated instructions (JSONL)')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output file for filtered instructions (text)')
    parser.add_argument('--seed_file', type=str, required=True,
                        help='File with seed instructions to include')
    parser.add_argument('--language', type=str, default='fi',
                        help='Language code for filtering (e.g., "fi" for Finnish)')
    
    args = parser.parse_args()
    
    process_output(args.input_file, args.output_file, args.seed_file, args.language)

if __name__ == "__main__":
    main()