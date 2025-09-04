#!/bin/bash

# SLURM Job Monitor
# Usage: ./monitor_slurm_job.sh --job_id JOB_ID [--check_interval SECONDS] [--prefix PREFIX]

job_id=""
check_interval=30
prefix="JOB"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --job_id)
      job_id="$2"
      shift 2
      ;;
    --check_interval)
      check_interval="$2"
      shift 2
      ;;
    --prefix)
      prefix="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$job_id" ]]; then
  echo "Usage: $0 --job_id JOB_ID [--check_interval SECONDS] [--prefix PREFIX]"
  exit 1
fi

echo "$prefix: Monitoring job $job_id"

while true; do
  state=$(sacct -j "$job_id" -o State --noheader 2>/dev/null | head -n1 | awk '{print $1}')
  rc=$?

  if [[ $rc -ne 0 ]]; then
    echo "$prefix: Error running sacct command. Check if SLURM is available."
    exit 1
  fi

  if [[ -z "$state" ]]; then
    echo "$prefix: Job $job_id not found in accounting yet, waiting..."
    sleep "$check_interval"
    continue
  fi

  case "$state" in
    PENDING|RUNNING|CONFIGURING|COMPLETING)
      echo "$prefix: Job $job_id still in progress with state: $state"
      sleep "$check_interval"
      ;;
    COMPLETED)
      echo "$prefix: Job $job_id completed successfully"
      echo "Waiting 5 seconds for file operations to complete..."
      sleep 5
      exit 0
      ;;
    *)
      echo "$prefix: Job $job_id failed with state: $state"
      exit 1
      ;;
  esac
done