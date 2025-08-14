import json
import random
from pathlib import Path
from datasets import load_dataset
from typing import List, Dict, Any, Optional

# Configuration variables
DATASET_1 = "databricks/databricks-dolly-15k"  # Dolly dataset from HuggingFace
DATASET_2 = "/scratch/project_462000353/posttraining_data/lmsys-chat-1m/train.jsonl"  # LMSYS Chat dataset
NUM_ROWS_2 = 150000 # Number of rows to sample from LMSYS
OUTPUT_FILE = "/scratch/project_462000353/adamhrin/dispatcher/examples/autoif/data/merged_dolly15k_lmsyschat150k.jsonl"

def load_dolly_data() -> List[Dict[str, Any]]:
    """Load all data from Databricks Dolly dataset."""
    data = []
    
    try:
        print(f"Loading from HuggingFace dataset: {DATASET_1}")
        dataset = load_dataset(DATASET_1)
        split_data = dataset['train']
        
        print(f"Loading all {len(split_data)} rows from Dolly dataset")
        
        for item in split_data:
            if 'instruction' in item and 'response' in item:
                # Filter out items with context
                context = item.get('context', '')
                if context and len(str(context).strip()) > 0:
                    continue
                    
                # Create metadata with all other fields
                metadata = {k: v for k, v in item.items() if k not in ['instruction', 'response']}

                data.append({
                    "queries": [str(item['instruction'])],
                    "responses": [str(item['response'])],
                    "metadata": metadata,
                    "source": DATASET_1
                })
    except Exception as e:
        print(f"Error loading dataset {DATASET_1}: {e}")
        return []
    
    print(f"Loaded {len(data)} valid rows from Dolly dataset")
    return data

def load_lmsys_data(num_rows: int) -> List[Dict[str, Any]]:
    """Load data from LMSYS Chat dataset."""
    data = []
    
    print(f"Loading from local file: {DATASET_2}")
    print(f"Target: {num_rows} valid rows")
    
    processed_lines = 0
    with open(DATASET_2, 'r', encoding='utf-8') as f:
        for line in f:
            # Stop when we have enough valid data
            if len(data) >= num_rows:
                break
                
            processed_lines += 1
            if processed_lines % 10000 == 0:
                print(f"Processed {processed_lines} lines, found {len(data)} valid rows...")
            
            try:
                item = json.loads(line.strip())
                
                # Filter for English language only
                if item.get('language') != 'English':
                    continue
                
                # Filter for first turns only
                if item.get('turn') != 1:
                    continue
                    
                # Check if conversation exists and has at least 2 messages
                conversation = item.get('conversation', [])
                if len(conversation) < 2:
                    continue
                
                # Find user query and assistant response
                user_content = None
                assistant_content = None
                
                for message in conversation:
                    if message.get('role') == 'user' and user_content is None:
                        user_content = message.get('content')
                    elif message.get('role') == 'assistant' and assistant_content is None:
                        assistant_content = message.get('content')
                    
                    # Break early if we found both
                    if user_content is not None and assistant_content is not None:
                        break
                
                if user_content and assistant_content:
                    # Create metadata with all other fields
                    metadata = { k: v for k, v in item.items() if k not in ['conversation', 'openai_moderation'] }

                    data.append({
                        "queries": [str(user_content)],
                        "responses": [str(assistant_content)],
                        "metadata": metadata,
                        "source": DATASET_2
                    })
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON line {processed_lines}: {e}")
                continue
    
    print(f"Processed {processed_lines} total lines")
    print(f"Loaded {len(data)} valid English rows from LMSYS dataset")
    
    if len(data) < num_rows:
        print(f"Warning: Only found {len(data)} valid rows, less than requested {num_rows}")
    
    return data

def merge_and_save_datasets(output_file: str = OUTPUT_FILE):
    """Merge the two datasets and save to JSONL file."""
    print("Starting data loading and merging process...")
    
    # Load data from both sources using specific loaders
    data_1 = load_dolly_data()
    data_2 = load_lmsys_data(NUM_ROWS_2)
    
    # Merge the datasets
    merged_data = data_1 + data_2
    
    # Shuffle the merged data
    random.shuffle(merged_data)
    
    # Save to JSONL file
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in merged_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Successfully merged {len(merged_data)} rows and saved to {output_file}")
    print(f"Dolly dataset contributed: {len(data_1)} rows")
    print(f"LMSYS dataset contributed: {len(data_2)} rows")
    
    return len(merged_data)

def main():
    """Main function to execute the data merging process."""
    # Set random seed for reproducibility
    random.seed(42)
    
    print("Configuration:")
    print(f"DATASET_1: {DATASET_1}")
    print(f"DATASET_2: {DATASET_2}")
    print(f"NUM_ROWS_2: {NUM_ROWS_2}")
    print("Dolly: using all data from 'instruction' and 'response' fields")
    print("LMSYS: using first user/assistant pair from 'conversation', English only")
    print("-" * 50)
    
    # Execute the merging process
    total_rows = merge_and_save_datasets()
    
    if total_rows > 0:
        print("\nData merging completed successfully!")
    else:
        print("\nWarning: No data was merged!")

if __name__ == "__main__":
    main()