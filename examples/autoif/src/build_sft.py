#!/usr/bin/env python3
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("--output", required=True)
    parser.add_argument("--score_threshold", type=int, required=True)
    args = parser.parse_args()
    
    with open(args.input_file, 'r') as f_in, open(args.output, 'w') as f_out:
        for line in f_in:
            data = json.loads(line.strip())
            score = data.get('score')
            
            if score is not None and score >= args.score_threshold:
                output = {'messages': data['messages']}
                f_out.write(json.dumps(output) + '\n')


if __name__ == "__main__":
    main()
