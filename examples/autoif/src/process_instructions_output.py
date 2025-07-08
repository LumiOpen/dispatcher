import argparse
import json
import re
# from huggingface_hub import InferenceClient

#language identifier
from huggingface_hub import hf_hub_download
import fasttext

model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin")   
model_cis_lmu = fasttext.load_model(model_path)

GLOT_LANG_DICT = {
    'bul': 'bg',  # Bulgarian
    'ces': 'cs',  # Czech
    'dan': 'da',  # Danish
    'deu': 'de',  # German
    'ell': 'el',  # Greek
    'eng': 'en',  # English
    'spa': 'es',  # Spanish
    'est': 'et',  # Estonian
    'ekk': 'et',  # Standard Estonian
    'fin': 'fi',  # Finnish
    'fra': 'fr',  # French
    'gle': 'ga',  # Irish
    'hrv': 'hr',  # Croatian
    'hun': 'hu',  # Hungarian
    'ita': 'it',  # Italian
    'lit': 'lt',  # Lithuanian
    'lav': 'lv',  # Latvian
    'lvs': 'lv',  # Standard Latvian
    'mlt': 'mt',  # Maltese
    'nld': 'nl',  # Dutch
    'pol': 'pl',  # Polish
    'por': 'pt',  # Portuguese
    'ron': 'ro',  # Romanian
    'slk': 'sk',  # Slovak
    'slv': 'sl',  # Slovenian
    'swe': 'sv'   # Swedish
}

def detect_language_glotlid(text,model=model_cis_lmu):
    """Given a text, it returns the Glotlid prediction as NLLB language code, e.g., Latn-eng
    """
    lang_code, score = model.predict(text)
    lang_code = lang_code[0].replace("__label__","").replace("_Latn","")
    return lang_code

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
                    try:
                        lang_code = detect_language_glotlid(instruction, model_cis_lmu)
                        if GLOT_LANG_DICT[lang_code] == language:
                            # Add to set to track duplicates
                            clean_instructions.add(instruction)
                            # Write to file directly
                            out_f.write(instruction + '\n')
                    except Exception as e:
                        print(f'Language detection error: {e}')

                    # # Add to set to track duplicates
                    # clean_instructions.add(instruction)
                    # # Write to file directly
                    # out_f.write(instruction + '\n')
    
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