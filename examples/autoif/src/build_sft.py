#!/usr/bin/env python3
import argparse
import json
import os
import random


def count_valid_entries(input_file, score_threshold):
    """Count total number of valid entries in the input file."""
    count = 0
    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            score = data.get('score')
            if score is not None and score >= score_threshold:
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("--output_dir", required=True, help="Directory to save the output files.")
    parser.add_argument("--score_threshold", type=int, default=4, help="Minimum score threshold for filtering messages.")
    parser.add_argument("--max_train_outrows", type=int, default=30000, help="Maximum number of rows in the train.jsonl file.")
    parser.add_argument("--test", action="store_true", help="If set, remaining entries after max_train_outrows will be written to test.jsonl.")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Count total valid entries first
    print("Counting valid entries...")
    total_valid = count_valid_entries(args.input_file, args.score_threshold)
    print(f"Found {total_valid} valid entries")
    
    if total_valid == 0:
        print("No valid entries found. Exiting.")
        return
    
    # Determine sampling strategy
    train_target = min(args.max_train_outrows, total_valid)
    test_target = max(0, total_valid - args.max_train_outrows) if args.test else 0
    
    print(f"Target: {train_target} train entries, {test_target} test entries")
    
    # Calculate sampling probabilities
    train_prob = train_target / total_valid
    
    # Prepare output file paths
    train_file = os.path.join(args.output_dir, "train.jsonl")
    test_file = os.path.join(args.output_dir, "test.jsonl") if args.test else None
    
    # Process file with streaming - no memory storage
    train_written = 0
    test_written = 0
    valid_seen = 0
    
    with open(args.input_file, 'r') as f_in, \
         open(train_file, 'w') as f_train, \
         (open(test_file, 'w') if test_file else None) as f_test:
        
        for line in f_in:
            data = json.loads(line.strip())
            score = data.get('score')
            
            if score is not None and score >= args.score_threshold:
                output = {
                    'messages': data['messages'], 
                    'query_source': data['query_metadata']['source'] if 'query_metadata' in data and 'source' in data['query_metadata'] else ''
                }
                valid_seen += 1
                
                # Simple deterministic split: first train_target go to train, rest to test
                if train_written < train_target:
                    # Use random sampling to get exactly train_target entries
                    # Probability of selecting this entry for train
                    remaining_valid = total_valid - valid_seen + 1
                    remaining_train_needed = train_target - train_written
                    
                    if remaining_train_needed >= remaining_valid:
                        # Must take this entry for train
                        f_train.write(json.dumps(output) + '\n')
                        train_written += 1
                    else:
                        # Random selection
                        prob = remaining_train_needed / remaining_valid
                        if random.random() < prob:
                            f_train.write(json.dumps(output) + '\n')
                            train_written += 1
                        elif f_test:
                            f_test.write(json.dumps(output) + '\n')
                            test_written += 1
                else:
                    # Train quota filled, everything goes to test
                    if f_test:
                        f_test.write(json.dumps(output) + '\n')
                        test_written += 1
    
    # Print summary
    print(f"Written {train_written} entries to {train_file}")
    if args.test:
        print(f"Written {test_written} entries to {test_file}")
    print(f"Total valid entries processed: {valid_seen}")


if __name__ == "__main__":
    main()
