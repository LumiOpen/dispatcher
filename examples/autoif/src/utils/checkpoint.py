#!/usr/bin/env python3
"""
Checkpoint management utilities.
Provides functions to check and modify checkpoint state.
"""
import datetime
import os
import sys

def is_step_completed(checkpoint_file, step_name):
    """Check if a step has been completed."""
    if not os.path.exists(checkpoint_file):
        return False
    
    with open(checkpoint_file, 'r') as f:
        for line in f:
            if line.strip().endswith(f"- {step_name}"):
                return True
    return False

def mark_step_completed(checkpoint_file, step_name):
    """Mark a step as completed in the checkpoint file."""
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(checkpoint_file, 'a') as f:
        f.write(f"{timestamp} - {step_name}\n")

def get_last_job_id(checkpoint_file, job_id_prefix):
    """Get the last job ID with the given prefix from checkpoint file."""
    if not os.path.exists(checkpoint_file):
        return None
    
    job_ids = []
    with open(checkpoint_file, 'r') as f:
        for line in f:
            if line.startswith(f"{job_id_prefix}="):
                job_ids.append(line.strip().split('=')[1])
    
    return job_ids[-1] if job_ids else None

def save_job_id(checkpoint_file, job_id_prefix, job_id):
    """Save a job ID to the checkpoint file."""
    with open(checkpoint_file, 'a') as f:
        f.write(f"{job_id_prefix}={job_id}\n")

def main():
    """CLI interface for checkpoint operations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Checkpoint management")
    parser.add_argument("--checkpoint_file", required=True, help="Checkpoint file path")
    subparsers = parser.add_subparsers(dest="command", help="Command to perform")
    
    # Check if step completed
    check_parser = subparsers.add_parser("check", help="Check if step is completed")
    check_parser.add_argument("--step", required=True, help="Step name to check")
    
    # Mark step completed
    mark_parser = subparsers.add_parser("mark", help="Mark step as completed")
    mark_parser.add_argument("--step", required=True, help="Step name to mark")
    
    # Get job ID
    get_parser = subparsers.add_parser("get-job", help="Get job ID")
    get_parser.add_argument("--prefix", required=True, help="Job ID prefix")
    
    # Save job ID
    save_parser = subparsers.add_parser("save-job", help="Save job ID")
    save_parser.add_argument("--prefix", required=True, help="Job ID prefix")
    save_parser.add_argument("--job-id", required=True, help="Job ID to save")
    
    args = parser.parse_args()
    
    if args.command == "check":
        result = is_step_completed(args.checkpoint_file, args.step)
        sys.exit(0 if result else 1)
    
    elif args.command == "mark":
        mark_step_completed(args.checkpoint_file, args.step)
        print(f"Marked step {args.step} as completed")
    
    elif args.command == "get-job":
        job_id = get_last_job_id(args.checkpoint_file, args.prefix)
        if job_id:
            print(job_id)
            sys.exit(0)
        else:
            print(f"No job ID found with prefix {args.prefix}", file=sys.stderr)
            sys.exit(1)
    
    elif args.command == "save-job":
        save_job_id(args.checkpoint_file, args.prefix, args.job_id)
        print(f"Saved job ID {args.job_id} with prefix {args.prefix}")

if __name__ == "__main__":
    main()