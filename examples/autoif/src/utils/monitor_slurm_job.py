#!/usr/bin/env python3
"""
SLURM Job Monitor

This script monitors a SLURM job and reports its status. It checks if the job is 
completed successfully, has failed, or is still running.

Usage:
  python monitor_slurm_job.py --job_id JOB_ID [--check_interval SECONDS] [--prefix PREFIX]
"""

import argparse
import subprocess
import time
import sys

def run_command(command):
    """Run a shell command and return its output."""
    # compatible with Python 3.6
    result = subprocess.run(
        command, 
        shell=True, 
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True  # This is equivalent to text=True in Python 3.7+
    )
    return result.stdout.strip(), result.returncode

def get_job_state(job_id):
    """Get the state of a SLURM job."""
    stdout, _ = run_command(f"sacct -j {job_id} -o State --noheader")
    if not stdout:
        return None
    
    # Get the first state for the main job (not steps)
    state = stdout.split('\n')[0].strip()
    return state

def main():
    parser = argparse.ArgumentParser(description="Monitor SLURM job status")
    parser.add_argument("--job_id", required=True, help="SLURM job ID")
    parser.add_argument("--check_interval", type=int, default=30,
                        help="Check interval in seconds (default: 30)")
    parser.add_argument("--prefix", default="JOB", help="Prefix for log messages")
    args = parser.parse_args()
    
    job_id = args.job_id
    prefix = args.prefix
    
    print(f"{prefix}: Monitoring job {job_id}")
    
    # Wait for job completion
    while True:
        job_state = get_job_state(job_id)
        
        if job_state is None:
            print(f"{prefix}: Job {job_id} not found in accounting yet, waiting...")
            time.sleep(args.check_interval)
            continue
            
        if job_state in ["PENDING", "RUNNING", "CONFIGURING", "COMPLETING"]:
            print(f"{prefix}: Job {job_id} still in progress with state: {job_state}")
            time.sleep(args.check_interval)
            continue
            
        if job_state != "COMPLETED":
            print(f"{prefix}: Job {job_id} failed with state: {job_state}")
            sys.exit(1)
            
        print(f"{prefix}: Job {job_id} completed successfully")
        # Wait for file operations to finish
        print(f"Waiting {args.check_interval} seconds for file operations to complete...")
        time.sleep(args.check_interval)
        sys.exit(0)

if __name__ == "__main__":
    main()