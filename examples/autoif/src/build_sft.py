#!/usr/bin/env python3
import argparse
import json
import os
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True, help="Input JSONL file with scored responses")
    parser.add_argument("--output-dir", required=True, help="Directory to save the output files.")
    parser.add_argument("--score-threshold", type=int, default=4, help="Minimum score threshold for filtering messages.")
    parser.add_argument("--max-train-outrows", type=int, default=30000, help="Maximum number of rows in the train.jsonl file.")
    parser.add_argument("--test", action="store_true", help="If set, remaining entries after max_train_outrows will be written to test.jsonl.")
    parser.add_argument("--max-test-outrows", type=int, default=3000, help="Maximum number of rows in the test.jsonl file.")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare output file paths
    train_file = os.path.join(args.output_dir, "train.jsonl")
    test_file = os.path.join(args.output_dir, "test.jsonl") if args.test else None
    
    # Process file with streaming - no memory storage
    train_written = 0
    test_written = 0
    valid_seen = 0
    turn_counts = {}  # Track count of samples by number of turns
    
    with open(args.input_file, 'r') as f_in, \
         open(train_file, 'w') as f_train:
        
        if test_file:
            f_test = open(test_file, 'w')
        else:
            f_test = None
        
        try:
            for line in f_in:
                data = json.loads(line.strip())
                scores = data.get('scores')

                if scores is not None and isinstance(scores, list) and all(s >= args.score_threshold for s in scores):
                    # Count number of turns (user messages)
                    messages = data.get('messages', [])
                    num_turns = sum(1 for msg in messages if msg.get('role') == 'user')
                    
                    # Track turn count
                    turn_counts[num_turns] = turn_counts.get(num_turns, 0) + 1
                    
                    output = {
                        'messages': messages, 
                        'query_source': data['query_metadata']['source'] if 'query_metadata' in data and 'source' in data['query_metadata'] else ''
                    }
                    valid_seen += 1
                    
                    # Simple sequential assignment: first max_train_outrows go to train, rest to test
                    if train_written < args.max_train_outrows:
                        f_train.write(json.dumps(output, ensure_ascii=False) + '\n')
                        train_written += 1
                    else:
                        # Train quota filled, everything goes to test
                        if f_test and test_written < args.max_test_outrows:
                            f_test.write(json.dumps(output, ensure_ascii=False) + '\n')
                            test_written += 1
        finally:
            if f_test:
                f_test.close()
    
    # Print summary
    print(f"Written {train_written} entries to {train_file}")
    if args.test:
        print(f"Written {test_written} entries to {test_file}")
    print(f"Total valid entries processed: {valid_seen}")
    
    # Report turn count statistics
    print("\nTurn count statistics:")
    for num_turns in sorted(turn_counts.keys()):
        count = turn_counts[num_turns]
        print(f"  {num_turns} turn(s): {count} samples")


if __name__ == "__main__":
    main()
