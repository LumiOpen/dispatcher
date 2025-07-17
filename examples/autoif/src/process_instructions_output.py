import argparse
import json
import re
import csv
from utils.lang_id import detect_language  


def process_output(input_file: str, output_file: str, seed_file: str, language: str = 'fi', max_instructions: int = 100) -> None:
    """
    De-duplicate instructions and filter by language. Do not include seed instructions in the output
    """

    seen_instructions = set()
    instruction_count = 0
    
    # Process instructions from model output
    try:
        with open(input_file, 'r') as f:
            results = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Input file {input_file} contains invalid JSON")
        exit(1)
    
    # Open CSV file for writing
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['id', 'instruction'])
        
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

                    # Break if we reach the maximum number of instructions
                    if instruction_count >= int(max_instructions):
                        print(f"Reached maximum number of instructions: {max_instructions}. Output {instruction_count} {language} instructions")
                        return

                    # Skip empty or very short instructions
                    if len(instruction) < 5:
                        continue
                    
                    # Skip duplicates
                    if instruction in seen_instructions:
                        continue
                    
                    # Check language
                    try:
                        lang_code1, lang_code2 = detect_language(instruction)
                        # lang_cde1 is the three-letter code, lang_code2 is the two-letter code
                        if lang_code1 == language or (lang_code2 is not None and lang_code2 == language):
                            print(f"Response language is {lang_code1} ({lang_code2}). Expected {language}.")
                            # Add to set to track duplicates
                            seen_instructions.add(instruction)
                            # Write immediately to CSV
                            writer.writerow([instruction_count, instruction])
                            instruction_count += 1
                    except Exception as e:
                        print(f'Language detection error: {e}')
    
    print(f'Output {instruction_count} {language} instructions')
    
def main():
    parser = argparse.ArgumentParser(description='Process and filter generated instructions')
    
    parser.add_argument('--input_file', type=str, required=True,
                        help='Input file with model-generated instructions (JSONL)')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output file for filtered instructions (CSV)')
    parser.add_argument('--seed_file', type=str, required=True,
                        help='File with seed instructions to include')
    parser.add_argument('--language', type=str, default='fi',
                        help='Language code for filtering (e.g., "fi" for Finnish)')
    parser.add_argument('--max_instructions', type=str, default=100,
                        help='Maximum number of instructions to output, default is 100.')
    
    args = parser.parse_args()
    
    process_output(args.input_file, args.output_file, args.seed_file, args.language, args.max_instructions)

if __name__ == "__main__":
    main()