#!/usr/bin/env python3
"""
Checkpoint management utilities.
Provides functions to check and modify checkpoint state.
"""
import datetime
import os
import sys
import json

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

def determine_continuation_point(checkpoint_file):
    """Determine which pipeline step to continue from based on the checkpoint file."""
    
    default_res = {"step": "AUG_START", "job_id": None}
    if not os.path.exists(checkpoint_file):
        return json.dumps(default_res)

    with open(checkpoint_file, 'r') as f:
        lines = f.readlines()
        last_lines = lines[-10:] if len(lines) >= 10 else lines

    last_checkpoint = None
    job_id = None
    
    # First, find the last checkpoint step
    for line in reversed(last_lines):
        line = line.strip()
        if not last_checkpoint and ' - ' in line:
            last_checkpoint = line.split(' - ')[1]
            break

    if not last_checkpoint:
        return json.dumps(default_res)

    # Determine which job ID prefix to look for based on the checkpoint
    job_id_prefix = None
    if last_checkpoint.startswith("AUG_"):
        job_id_prefix = "AUG_JOB_ID"
    elif last_checkpoint.startswith("VER_"):
        job_id_prefix = "VER_JOB_ID"
    elif last_checkpoint.startswith("RESP_"):
        job_id_prefix = "RESP_JOB_ID"
    
    # Now look for the latest job ID with the specific prefix
    # We need to search the entire file, not just the last 10 lines,
    # to handle cases where there are multiple job IDs due to restarts
    if job_id_prefix:
        job_id = get_last_job_id(checkpoint_file, job_id_prefix)

    transitions = {
        "AUG_PREPROCESSING": "AUG_INFERENCE",
        "AUG_INFERENCE_COMPLETE": "AUG_POSTPROCESS",
        "AUG_INFERENCE_FAILED": "AUG_INFERENCE",
        "AUG_POSTPROCESSING": "VER_START",
        "VER_PREPROCESSING": "VER_INFERENCE",
        "VER_INFERENCE_COMPLETE": "VER_CROSSVAL",
        "VER_INFERENCE_FAILED": "VER_INFERENCE",
        "VER_CROSS_VALIDATION": "CONCAT_START",
        "CONCAT_QUERIES_CONCATED": "RESP_START",
        "RESP_INFERENCE_COMPLETE": "SFT_START",
        "RESP_INFERENCE_FAILED": "RESP_START",
        "SFT_DATASET_BUILT": "COMPLETE"
    }

    # Special handling for SUBMITTED states - extract phase prefix to match pipeline case names
    if last_checkpoint.endswith("_SUBMITTED"):
        phase_prefix = last_checkpoint.split("_")[0]
        next_step = f"{phase_prefix}_MONITOR" if job_id else f"{phase_prefix}_INFERENCE"
        return json.dumps({"step": next_step, "job_id": job_id})

    next_step = transitions.get(last_checkpoint, "AUG_START")
    return json.dumps({"step": next_step, "job_id": job_id})

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
    
    # Get continuation point
    subparsers.add_parser("get-continuation", help="Get the next step to continue from")
    
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
    
    elif args.command == "get-continuation":
        continuation_json = determine_continuation_point(args.checkpoint_file)
        print(continuation_json)
        sys.exit(0)

if __name__ == "__main__":
    main()