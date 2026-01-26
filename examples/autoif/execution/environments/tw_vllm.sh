#!/bin/bash
#
# Interactive vLLM Server - Single Node Setup
#
# Usage: Run this script in an interactive SLURM allocation:
#   salloc --nodes=1 --gpus-per-node=4 --partition=amd-tw-verification ...
#   ./interactive_vllm_new.sh

# Set up logging
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_OUT="$LOG_DIR/${SLURM_JOB_ID:-$$}.out"
LOG_ERR="$LOG_DIR/${SLURM_JOB_ID:-$$}.err"

# Redirect all output to log files
exec > "$LOG_OUT" 2> "$LOG_ERR"

###############################################################################
# Configuration overrides (before sourcing launcher)
###############################################################################

# Qwen3 MoE: Use CK flash attention, enable AITER online tuning
export VLLM_USE_TRITON_FLASH_ATTN=0    # 0 = Use CK flash attention (faster)
export VLLM_ROCM_USE_AITER=0           # 1 = Use AITER CK kernels
# export AITER_ONLINE_TUNE=0             # 1 = Auto-tune MoE kernels for each shape

###############################################################################
# Source launcher (handles container setup, triton cache, AITER init)
###############################################################################

source /shared_silo/scratch/adamhrin@amd.com/dispatcher/examples/autoif/execution/environments/singularity_launcher.sh

###############################################################################
# Launch vLLM server
###############################################################################

MODEL=/shared_silo/scratch/models/Qwen3-30B-A3B
TP=4
MAX_MODEL_LEN=32768

echo "============================================"
echo "Starting vLLM server"
echo "  Model: $MODEL"
echo "  Tensor Parallel: $TP"
echo "  Max Model Len: $MAX_MODEL_LEN"
echo "============================================"

run_sing_bash "
    run_python -m vllm.entrypoints.openai.api_server \
        --model $MODEL \
        --host 0.0.0.0 \
        --port 8000 \
        --tensor-parallel-size $TP \
        --max-model-len $MAX_MODEL_LEN \
        --block-size 16 \
        --gpu-memory-utilization 0.95 \
        --quantization fp8
"
