#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --nodes=1
#SBATCH --partition=dev-g
#SBATCH --time=00-02:00:00
#SBATCH --ntasks-per-node=2
#SBATCH --mem=480G
#SBATCH --cpus-per-task=7
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --gpus-per-node=mi250:8
#SBATCH --account=project_462000963
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err


###
# configure the following.

INPUT_FILE=input.jsonl
OUTPUT_FILE=output.jsonl

# jq-like path string to find the prompt within the input jsonl row.
PROMPT_PATH='.messages[0].content'

# Prompting mode is "chat" or "completion"
MODE=chat
STOP_WORD=$'\n\n'  # $'' format allows escape chars to be interpreted.

# generation parameters
BATCH_SIZE=64       # number of prompts in a batch
NUM_GENERATIONS=1   # generations per prompt

# sampling params
MIN_P=0.05
TOP_P=1.00
TEMPERATURE=0.8


#
# If you are changing the model, be sure to update GPUS_PER_TASK and the
# sbatch --ntasks-per-node configuration appropriately.
# Typically on Lumi 70B = 4 GPUs, 34B = 2 GPUs, 8B = 1 GPU
# --ntasks-per-node should be int(8 / GPUS_PER_TASK)
#
MODEL=meta-llama/Llama-3.3-70B-Instruct
GPUS_PER_TASK=4     # enough for the model and large batch size
MAX_MODEL_LEN=16384 # only as much as you think you need for efficiency
MAX_TOKENS=4096     # max tokens to generate

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
LAUNCHER_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "$LAUNCHER_DIR/singularity_launcher.sh"

# Start dispatcher server in background
echo "Starting dispatcher server..."
run_sing_python -m dispatcher.server \
  --infile "$INPUT_FILE" \
  --outfile "$OUTPUT_FILE" \
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
# Environment is automatically set up when sourcing the launcher
srun -l bash -c "
  # Source launcher - auto-translates SLURM vars and sets up environment
  source \"${LAUNCHER_DIR}/singularity_launcher.sh\"
  
  run_sing_bash '
    set -euxo pipefail

    export HOME=/workspace
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

    echo \"Launching task LOCALID=\$LOCALID (global id: \$SLURM_PROCID) on GPUs \$HIP_VISIBLE_DEVICES (MASTER_PORT=\$MASTER_PORT, VLLM_PORT=\$VLLM_PORT)\"

    # Worker environment is automatically set up when sourcing inside container
    source \"\$HOME/singularity_launcher.sh\"

    # Run inference worker
    echo \"Starting inference worker...\"
    run_python /workspace/inference.py \
      --batch_size $BATCH_SIZE \
      --dispatcher_server $DISPATCHER_SERVER:$DISPATCHER_PORT \
      --prompt_path \"$PROMPT_PATH\" \
      --mode $MODE \
      --stop_word \"$STOP_WORD\" \
      --num_generations $NUM_GENERATIONS \
      --max_model_len $MAX_MODEL_LEN \
      --max_tokens $MAX_TOKENS \
      --min_p $MIN_P \
      --top_p $TOP_P \
      --temperature $TEMPERATURE \
      --tensor_parallel_size $GPUS_PER_TASK \
      --model_path $MODEL
  '
"

