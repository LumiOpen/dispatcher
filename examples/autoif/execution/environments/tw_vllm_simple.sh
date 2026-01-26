#!/bin/bash
#
# Simplified vLLM Server startup
#

cd /shared_silo/scratch/adamhrin@amd.com/dispatcher/examples/autoif

# Set up logging
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_OUT="$LOG_DIR/${SLURM_JOB_ID:-$$}.out"
LOG_ERR="$LOG_DIR/${SLURM_JOB_ID:-$$}.err"

# Redirect all output to log files
exec > "$LOG_OUT" 2> "$LOG_ERR"

# Container and paths
IMG="/shared_silo/scratch/containers/rocm_vllm_rocm7.0.0_vllm_0.11.1_20251103.sif"
LAUNCHER_HOME="/shared_silo/scratch/adamhrin@amd.com"

# Model config
MODEL=/shared_silo/scratch/models/Qwen3-30B-A3B
TP=4
MAX_MODEL_LEN=32768

echo "============================================"
echo "Starting vLLM server (simplified)"
echo "  Model: $MODEL"
echo "  Tensor Parallel: $TP"
echo "  Max Model Len: $MAX_MODEL_LEN"
echo "  Container: $IMG"
echo "============================================"

# Use --cleanenv to avoid picking up user's local packages
# Set PYTHONUSERBASE to empty to prevent user site-packages
singularity exec --rocm --cleanenv \
    -B /shared_silo/scratch/adamhrin@amd.com:/shared_silo/scratch/adamhrin@amd.com:rw \
    -B /shared_silo/scratch/models:/shared_silo/scratch/models:ro \
    -B "${PWD}:/workspace" \
    "$IMG" \
    bash -c '
        export HOME=/shared_silo/scratch/adamhrin@amd.com
        export HF_HOME=/shared_silo/scratch/adamhrin@amd.com/hf_cache
        export PYTHONUSERBASE=""
        export PYTHONNOUSERSITE=1
        export VLLM_USE_TRITON_FLASH_ATTN=0
        export VLLM_ROCM_USE_AITER=0
        
        python3 -m vllm.entrypoints.openai.api_server \
            --model '"$MODEL"' \
            --host 0.0.0.0 \
            --port 8000 \
            --tensor-parallel-size '"$TP"' \
            --max-model-len '"$MAX_MODEL_LEN"' \
            --block-size 16 \
            --gpu-memory-utilization 0.95
    '
