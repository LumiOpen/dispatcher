#!/bin/bash
###############################################################################
# Wait for the six translation dispatcher servers, then submit eight workers
# for each language. Marker files include the server instance identity so an
# old campaign cannot suppress a new submission.
###############################################################################

#SBATCH --job-name=disp-worker-gate
#SBATCH --account=project_462001516
#SBATCH --partition=small
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --open-mode=append
#SBATCH --output=logs/dispatcher-worker-gate-%j.out
#SBATCH --error=logs/dispatcher-worker-gate-%j.err

set -euo pipefail

WORK_DIR="${WORK_DIR:-$SLURM_SUBMIT_DIR}"
WORKER_LAUNCHER="${WORKER_LAUNCHER:-jobs/lumi/launch_dispatcher_worker_lumi.sh}"
WORKER_CONFIG="${WORKER_CONFIG:-configs/dispatcher-worker-translation-split-traces-lumi.conf}"
LAUNCHER_HF_HOME="${LAUNCHER_HF_HOME:-/scratch/project_462001516/users/adamhrin/hf-home}"
POLL_INTERVAL="${POLL_INTERVAL:-15}"
WORKER_COUNT="${WORKER_COUNT:-8}"
WORKER_TIME="${WORKER_TIME:-3-00:00:00}"

cd "$WORK_DIR"

if [[ "$WORKER_CONFIG" != /* ]]; then
  WORKER_CONFIG="$WORK_DIR/$WORKER_CONFIG"
fi
if [ ! -f "$WORKER_CONFIG" ]; then
  echo "ERROR: Worker config not found: $WORKER_CONFIG" >&2
  exit 1
fi

languages=(it nl pl ro sv uk)

active_worker_job() {
  local job_name=$1
  squeue -h -u "$USER" --name="$job_name" -o '%A' | head -n 1
}

submit_worker() {
  local language=$1
  local worker_index=$2
  local server_key=$3
  local address_file=".dispatcher-server-translation-split-traces-$language"
  local marker_file=".dispatcher-worker-$language-$worker_index.jobid"
  local job_name="disp-w-$language-$worker_index"

  if [ -s "$marker_file" ]; then
    local marker_job_id marker_server_key
    marker_job_id=$(sed -n '1p' "$marker_file")
    marker_server_key=$(sed -n '2p' "$marker_file")
    if [ "$marker_server_key" = "$server_key" ]; then
      echo "Worker already submitted for this server: language=$language index=$worker_index job=$marker_job_id"
      return
    fi
    echo "Replacing stale worker marker: language=$language index=$worker_index old_job=$marker_job_id"
  fi

  local existing_job_id
  existing_job_id=$(active_worker_job "$job_name")
  if [ -n "$existing_job_id" ]; then
    printf '%s\n%s\n' "$existing_job_id" "$server_key" > "$marker_file"
    echo "Adopted active worker: language=$language index=$worker_index job=$existing_job_id"
    return
  fi

  local job_id
  job_id=$(sbatch --parsable \
    --time="$WORKER_TIME" \
    --chdir="$WORK_DIR" \
    --job-name="$job_name" \
    --output="logs/dispatcher-worker-$language-$worker_index-%j.out" \
    --error="logs/dispatcher-worker-$language-$worker_index-%j.err" \
    --open-mode=append \
    --export="ALL,WORK_DIR=$WORK_DIR,LANGUAGE=$language,SERVER_ADDRESS_FILE=$address_file,LAUNCHER_HF_HOME=$LAUNCHER_HF_HOME" \
    "$WORKER_LAUNCHER" \
    "$WORKER_CONFIG")

  printf '%s\n%s\n' "$job_id" "$server_key" > "$marker_file"
  echo "Submitted worker: language=$language index=$worker_index job=$job_id"
}

while true; do
  remaining=0

  for language in "${languages[@]}"; do
    address_file=".dispatcher-server-translation-split-traces-$language"
    if [ ! -s "$address_file" ]; then
      echo "Waiting for $language server address file: $address_file"
      remaining=1
      continue
    fi

    address=$(<"$address_file")
    if ! curl -sf -o /dev/null --max-time 5 "http://$address/status"; then
      echo "Waiting for $language server health endpoint: http://$address/status"
      remaining=1
      continue
    fi

    echo "Server ready: language=$language address=$address"
    server_key="$address:$(stat -c '%Y' "$address_file")"
    for worker_index in $(seq 1 "$WORKER_COUNT"); do
      submit_worker "$language" "$worker_index" "$server_key"
    done
  done

  if [ "$remaining" -eq 0 ]; then
    echo "All $(( ${#languages[@]} * WORKER_COUNT )) worker jobs are active or submitted."
    exit 0
  fi

  sleep "$POLL_INTERVAL"
done
