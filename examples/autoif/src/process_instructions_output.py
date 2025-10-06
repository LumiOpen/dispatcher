import argparse
import json
import re
import csv
from utils.lang_id import detect_language  


def process_output(input_file: str, output_file: str, language: str = 'en', max_instructions: int = 100, seed_file: str = None) -> None:
    """
    De-duplicate instructions and filter by language. Do not include seed instructions in the output
    """

    seen_instructions = set()
    instruction_count = 0
    
    # Load keyword instructions from seed file if provided
    keyword_instructions = []
    if seed_file:
        try:
            with open(seed_file, 'r') as f:
                seed_data = json.load(f)
            
            # Extract keyword instructions from the new JSON format
            if "keyword_instructions" in seed_data:
                keyword_instructions = [
                    {'instruction': instruction, 'category': 'keyword'} 
                    for instruction in seed_data["keyword_instructions"]
                ]
                print(f"Loaded {len(keyword_instructions)} keyword instructions from {seed_file}")
            else:
                print(f"Warning: No keyword_instructions found in {seed_file}")
        except FileNotFoundError:
            print(f"Warning: Seed file {seed_file} not found, continuing without keyword instructions")
        except json.JSONDecodeError:
            print(f"Warning: Invalid JSON in seed file {seed_file}, continuing without keyword instructions")
    
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
        # Write header with new keyword and category columns
        writer.writerow(['id', 'instruction', 'category'])
        
        # First, add keyword instructions to the CSV
        for keyword_data in keyword_instructions:
            if instruction_count >= int(max_instructions):
                break
            
            keyword_instruction = keyword_data['instruction']
            keyword_category = keyword_data['category']
            
            # Skip duplicates
            if keyword_instruction in seen_instructions:
                print(f"Skipping duplicate keyword instruction: <instruction>{keyword_instruction}</instruction>")
                continue
                
            seen_instructions.add(keyword_instruction)
            writer.writerow([instruction_count, keyword_instruction, keyword_category])
            instruction_count += 1
            print(f"Added keyword instruction {instruction_count}: {keyword_instruction}")
        
        # Calculate remaining instructions and per-category limit
        remaining_instructions = int(max_instructions) - instruction_count
        instructions_per_category = remaining_instructions // len(results) if len(results) > 0 else 0
        print(f"Added {instruction_count} keyword instructions. Remaining: {remaining_instructions}")
        print(f"Will add up to {instructions_per_category} instructions per category across {len(results)} categories")
        
        # Then process regular augmented instructions
        for result in results:
            # Get category from original data if it exists
            category = result.get('original', {}).get('category', None)
            cat_instr_count = 0
            for response in result.get('responses', []):
                # Extract instructions
                instructions = re.findall(r'^\d+\.?\s+(.*)', response, re.MULTILINE)
                
                # If no numbered instructions found, try line-by-line
                if not instructions:
                    instructions = [line.strip() for line in response.split('\n') 
                                    if line.strip() and not line.strip().startswith('#')]

                print(f"Processing {len(instructions)} raw instructions from response {'for category ' + category if category else ''}")
                
                for instruction in instructions:
                    instruction = instruction.strip()

                    if cat_instr_count >= instructions_per_category:
                        print(f"Reached maximum of {instructions_per_category} instructions for category {category}, moving to next category")
                        break

                    # Break if we reach the maximum number of instructions
                    if instruction_count >= int(max_instructions):
                        print(f"Reached maximum number of instructions: {max_instructions}. Output {instruction_count} {language} instructions")
                        return

                    # Skip empty or very short instructions
                    if len(instruction) < 5:
                        print(f"Skipping: len(instruction) < 5: <instruction>{instruction}</instruction>")
                        continue
                    
                    # Skip duplicates
                    if instruction in seen_instructions:
                        print(f"Skipping: instruction already seen: <instruction>{instruction}</instruction>")
                        continue
                    
                    # Check language
                    try:
                        lang_code1, lang_code2 = detect_language(instruction)
                        # lang_cde1 is the three-letter code, lang_code2 is the two-letter code
                        if lang_code1 == language or (lang_code2 is not None and lang_code2 == language):
                            print(f"OK. Response language is {lang_code1} ({lang_code2}). Expected {language}.")
                            # Add to set to track duplicates
                            seen_instructions.add(instruction)
                            # Write to CSV with keyword=False and category
                            writer.writerow([instruction_count, instruction, category])
                            cat_instr_count += 1
                            instruction_count += 1
                    except Exception as e:
                        print(f'Language detection error: {e}')
    
    print(f'Output {instruction_count} {language} instructions (including {len([inst for inst in keyword_instructions if inst["instruction"] in seen_instructions])} keyword instructions)')

def main():
    parser = argparse.ArgumentParser(description='Process and filter generated instructions')
    
    parser.add_argument('--input_file', type=str, required=True,
                        help='Input file with model-generated instructions (JSONL)')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output file for filtered instructions (CSV)')
    parser.add_argument('--language', type=str, default='fi',
                        help='Language code for filtering (e.g., "fi" for Finnish)')
    parser.add_argument('--max_instructions', type=str, default=100,
                        help='Maximum number of instructions to output, default is 100.')
    parser.add_argument('--seed_file', type=str, default=None,
                        help='Optional seed file (categorised_instructions.json) containing keyword instructions to include in output')
    
    args = parser.parse_args()
    
    process_output(args.input_file, args.output_file, args.language, args.max_instructions, args.seed_file)

if __name__ == "__main__":
    main()