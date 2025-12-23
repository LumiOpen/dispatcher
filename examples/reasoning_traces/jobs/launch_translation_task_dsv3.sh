#!/bin/bash
#SBATCH --job-name=translation
#SBATCH --nodes=8
#SBATCH --partition=amd-tw-verification
#SBATCH --time=02-00:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --mem=480G
#SBATCH --cpus-per-task=7
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --gpus-per-node=8
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err


###
# configure the following.

INPUT_FILE=/shared_silo/scratch/datasets/Llama-Nemotron-SFT-math-100k.jsonl
OUTPUT_FILE=/shared_silo/scratch/adamhrin@amd.com/dispatcher/examples/reasoning_traces/data/Llama-Nemotron-SFT-math-100k_translated_DeepSeek-V3_fi.jsonl
TASK=tasks.reasoning_translation_task.ReasoningTranslationTask
export LANGUAGE=fi

# generation parameters
# These should be tuned so that you do not overload your backend vllm server,
# or run into any timeouts.  timeouts greatly affect the efficiency of the
# workflow.
# Allow environment overrides for easy sweeps via: sbatch --export=ALL,WORKERS=160,MAX_NUM_SEQS=192,...
WORKERS=${WORKERS:-96}          # number of simultaneous backend requests (per vLLM server)
BATCH_SIZE=${BATCH_SIZE:-1}      # amount of work to request from dispatcher. 1 is usually fine.

# Timeouts are safety valves and you should not hit them in the normal course
# of your workflow.  if you do, it suggests you need to change something about
# your configuration--tasks are usually written to expect success.
REQUEST_TIMEOUT=4800 # adjust as needed for your task so that you do not hit
WORK_TIMEOUT=7200   # time for dispatcher to give up on a work item and reissue it.  ideally this should never be hit.
STARTUP_TIMEOUT=7200 # time for dispatcher to give up on a startup and reissue it.  ideally this should never be hit.

#
# If you are changing the model, be sure to update GPUS_PER_TASK and the
# sbatch --ntasks-per-node configuration appropriately.
# Typically on Lumi 70B = 4 GPUs, 34B = 2 GPUs, 8B = 1 GPU
# --ntasks-per-node should be int(8 / GPUS_PER_TASK)
#
export MODEL=/shared_silo/scratch/models/DeepSeek-V3
GPUS_PER_TASK=8     # enough for the model and large batch size
# based on https://rocm.docs.amd.com/en/docs-7.0-docker/benchmark-docker/inference-vllm-deepseek-r1-fp8.html
MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536} # Must be >= the input + the output lengths.
MAX_SEQ_LEN_TO_CAPTURE=${MAX_SEQ_LEN_TO_CAPTURE:-65536}  # Beneficial to set this to max_model_len.
# Keep max-num-seqs close to WORKERS (or slightly above) to avoid surprising KV pressure.
MAX_NUM_SEQS=${MAX_NUM_SEQS:-128}
# Offline throughput: try a larger token budget (chunked prefill is enabled in your vLLM build).
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-262144} # Smaller values may result in better TTFT but worse TPOT / throughput.

# end configuration
###################

set -euxo pipefail

# dispatcher server will run on the first node, before we launch the worker
# tasks.
export DISPATCHER_SERVER=$(hostname)
export DISPATCHER_PORT=9999

###############################################################################
# Load launcher library - environment is automatically set up when sourced
###############################################################################
source /shared_silo/scratch/adamhrin@amd.com/dispatcher/examples/reasoning_traces/singularity_launcher.sh

# Start dispatcher server in background
echo "Starting dispatcher server..."
run_sing_python -m dispatcher.server \
  --infile "$INPUT_FILE" \
  --outfile "$OUTPUT_FILE" \
  --work-timeout "$WORK_TIMEOUT" \
  --host 0.0.0.0 \
  --port "$DISPATCHER_PORT" &
srv_pid=$!

# Wait for server to be ready
echo "Waiting for dispatcher server on $DISPATCHER_SERVER:$DISPATCHER_PORT..."
for i in $(seq 1 120); do
  (echo >/dev/tcp/$DISPATCHER_SERVER/${DISPATCHER_PORT}) >/dev/null 2>&1 && break || true
  sleep 1
done
(echo >/dev/tcp/$DISPATCHER_SERVER/${DISPATCHER_PORT}) >/dev/null 2>&1 || {
  echo "[ERROR] dispatcher server did not start"
  exit 1
}
echo "Server is up."

# Launch workers in containers
# SLURM variables are automatically translated to SINGULARITYENV_* by the launcher
# Environment is automatically set up by run_sing_bash
srun -l bash -c "
  run_sing_bash '
    set -euxo pipefail

    LOCALID=\${SLURM_LOCALID:-0}

    # Compute GPU allocation (use LOCALID for per-node GPU assignment)
    start_gpu=\$(( LOCALID * $GPUS_PER_TASK ))
    GPU_IDS=\"\"
    for (( i=0; i<$GPUS_PER_TASK; i++ )); do
      if [ -z \"\$GPU_IDS\" ]; then GPU_IDS=\"\$(( start_gpu + i ))\"; else GPU_IDS=\"\${GPU_IDS},\$(( start_gpu + i ))\"; fi
    done
    export HIP_VISIBLE_DEVICES=\"\$GPU_IDS\"

    # Use LOCALID for ports to ensure uniqueness across tasks on the same node
    export MASTER_ADDR=\${MASTER_ADDR:-127.0.0.1}
    export MASTER_PORT=\$(( 7000 + LOCALID ))
    export VLLM_PORT=\$(( 8000 + LOCALID * 100 ))

    export LANGUAGE=$LANGUAGE
    export MODEL=$MODEL

    echo \"Launching task LOCALID=\$LOCALID (global id: \$SLURM_PROCID) on GPUs \$HIP_VISIBLE_DEVICES (MASTER_PORT=\$MASTER_PORT, VLLM_PORT=\$VLLM_PORT)\"

    # Run task manager worker
    # Based on https://rocm.docs.amd.com/en/docs-7.0-docker/benchmark-docker/inference-vllm-deepseek-r1-fp8.html
    # Note: Using --kv-cache-dtype fp8 with DeepSeek may cause accuracy issues
    echo \"Starting dispatcher task manager...\"
    run_python -m dispatcher.taskmanager.cli \
      --dispatcher $DISPATCHER_SERVER:$DISPATCHER_PORT \
      --task $TASK \
      --batch-size $BATCH_SIZE \
      --workers $WORKERS \
      --max-model-len $MAX_MODEL_LEN \
      --tensor-parallel $GPUS_PER_TASK \
      --model $MODEL \
      --port \$VLLM_PORT \
      --request-timeout $REQUEST_TIMEOUT \
      --startup-timeout $STARTUP_TIMEOUT \
      --silence-vllm-logs \
      --vllm-extra-args \"--swap-space 0 --max-num-seqs ${MAX_NUM_SEQS} --no-enable-prefix-caching --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} --block-size 1 --gpu-memory-utilization 0.95 --async-scheduling --quantization fp8\"
  '
"

