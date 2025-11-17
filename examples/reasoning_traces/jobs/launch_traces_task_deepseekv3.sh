#!/bin/bash
#SBATCH --job-name=dispatcher-deepseek
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=250G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --account=amd-silo-mdl
#SBATCH --partition=amd-silo-mdl
#SBATCH --nodelist=chi-mi300x-025

mkdir -p logs

# echo "Testing docker run..."
# docker run --rm ubuntu:20.04 bash -c "echo 'Hello from inside container'; sleep 30"
# echo "Done"

MODEL_PATH=/vfs_mount/models/DeepSeek-V3
CONTAINER_NAME=${SLURM_JOB_NAME}-${USER}-${SLURM_JOB_ID}-slurmdocker # this is what slurmdocker names the container

# Step 1: Start the vLLM server container in detached mode
slurmdocker -d \
  -v $SCRATCH:/workspace \
  -v /vfs/silo/basemodels-team/:/vfs_mount \
  --ipc=host \
  -w /workspace \
  rocm/vllm-dev:nightly_0624_rc2_0624_rc2_20250620 \
  bash -c "
    set -x
    echo 'Inside container at' \$(date)
    export SAFETENSORS_FAST_GPU=1
    export VLLM_ROCM_USE_AITER=1
    export VLLM_USE_V1=1
    vllm serve $MODEL_PATH \
      -tp 8 \
      --max-model-len 65536 \
      --block-size 1 \
      --max_seq_len_to_capture 65536 \
      --no-enable-prefix-caching \
      --max-num-batched-tokens 65536 \
      --port 8000 \
      --trust-remote-code \
      --gpu-memory-utilization 0.85 \
      > /workspace/adamhrin/dispatcher/logs/vllm.log 2>&1
  "
# Step 2: Wait for server to be ready
echo "[$(date)] Waiting for vLLM server to start..."
echo "Check vLLM logs with tail -f $SCRATCH/adamhrin/dispatcher/logs/vllm.log"
# Wait until vllm is up
while true; do
    if docker exec $CONTAINER_NAME curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "vLLM server is ready!"
        break
    fi
    sleep 30
done

docker logs $CONTAINER_NAME | tail -n 20

# Step 3: Launch your client in the same container 
echo "[$(date)] Launching client inside container..."
docker exec $CONTAINER_NAME bash -c "
    set -x
    cd /workspace/adamhrin/dispatcher2/examples/translation

    python3 -u pipeline.py --config configs/deepseekv3_traces.yaml
"

# Step 4: stop docker container
# slurmdocker stop $CONTAINER_NAME
