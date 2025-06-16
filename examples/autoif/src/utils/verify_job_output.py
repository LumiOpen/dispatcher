#!/usr/bin/env python3
"""
Job Output Verifier

This script verifies that the output files of a job meet certain criteria,
such as existence and having the expected number of lines.

Usage:
  python verify_job_output.py --input_file FILE --output_file FILE
                             [--required_completion PERCENT]
                             [--log_file FILE] [--prefix PREFIX]
"""

import argparse
import os
import sys

def file_line_count(file_path):
    """Count the number of lines in a file."""
    if not os.path.exists(file_path):
        return 0
    
    with open(file_path, 'r') as f:
        return sum(1 for _ in f)

def main():
    parser = argparse.ArgumentParser(description="Verify job output files")
    parser.add_argument("--input_file", required=True, help="Input file path")
    parser.add_argument("--output_file", required=True, help="Output file path")
    parser.add_argument("--required_completion", type=int, default=100, 
                        help="Required completion percentage (default: 100)")
    parser.add_argument("--log_file", help="Path to job log file")
    parser.add_argument("--prefix", default="JOB", help="Prefix for log messages")
    args = parser.parse_args()
    
    prefix = args.prefix
    
    print(f"{prefix}: Verifying job outputs...")
    
    # Check if output file exists
    if not os.path.exists(args.output_file):
        print(f"ERROR: Output file {args.output_file} missing!")
        sys.exit(1)
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"ERROR: Input file {args.input_file} missing!")
        sys.exit(1)
    
    # Compare line counts
    input_lines = file_line_count(args.input_file)
    output_lines = file_line_count(args.output_file)
    
    print(f"Verification: Found {output_lines} outputs for {input_lines} inputs")
    
    if input_lines == 0:
        print(f"ERROR: Input file {args.input_file} is empty!")
        sys.exit(1)
    
    completion_percentage = (output_lines * 100) // input_lines
    
    if input_lines == output_lines:
        print(f"All tasks were successfully processed!")
    else:
        print(f"WARNING: Not all tasks were processed! ({output_lines}/{input_lines}, {completion_percentage}%)")
        
        if completion_percentage < args.required_completion:
            print(f"Completion percentage {completion_percentage}% is below required {args.required_completion}%")
            if args.log_file:
                print(f"Check log file for errors: {args.log_file}")
            sys.exit(1)
        else:
            print(f"More than {args.required_completion}% of tasks completed. Continuing with available results.")
    
    print(f"{prefix}: Job verification completed successfully!")
    sys.exit(0)

if __name__ == "__main__":
    main()