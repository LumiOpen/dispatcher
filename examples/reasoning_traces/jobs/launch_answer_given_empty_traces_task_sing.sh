#!/bin/bash
#SBATCH --job-name=ans_traces
#SBATCH --nodes=1
#SBATCH --partition=dev-g
#SBATCH --time=02:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --mem=480G
#SBATCH --cpus-per-task=7
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --gpus-per-node=4
#SBATCH --account=project_462000963
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#
# This script defines a 100% clean PATH inside the container
# to prevent host environment pollution, which was causing
# the 'ninja' ModuleNotFoundError.
#
set -euxo pipefail

### config
export LANGUAGE="${1:-fi}"
export MODEL="${2:-Qwen/Qwen3-30B-A3B-Thinking-2507}"
MODEL_NAME="$(basename "$MODEL")"
INPUT_FILE="/scratch/project_462000353/adamhrin/dispatcher/examples/reasoning_traces/data/default-train-sample-100_translations_DeepSeek-V3_fi.jsonl"

DATADIR="$(dirname "$INPUT_FILE")"
FILE_NAME="$(basename "$INPUT_FILE" .jsonl)"
OUTPUT_FILE="${DATADIR}/${FILE_NAME}_answers_given_empty_traces_${MODEL_NAME}_${LANGUAGE}.jsonl"
TASK="tasks.answering_given_empty_traces_task.AnsweringGivenEmptyTracesTask"

WORKERS=16
BATCH_SIZE=1
WORK_TIMEOUT=3600
GPUS_PER_TASK=4
MAX_MODEL_LEN=16384
STARTUP_TIMEOUT=7200
### end config

mkdir -p logs pythonuserbase

# Caches MUST be on a writable, bound path. Use your ...963 project.
export HF_HOME="/scratch/project_462000963/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export TORCHINDUCTOR_CACHE="/scratch/project_462000963/torch_inductor_cache"
mkdir -p "$HF_HOME" "$TORCHINDUCTOR_CACHE"

# Container and Python paths
export IMG="/scratch/project_462000353/containers/vllm_v10.1.1.sif"
export PYEXEC_IN_IMG="/opt/miniconda3/envs/pytorch/bin/python"
export PIP_IN_IMG="$PYEXEC_IN_IMG -m pip"

# Compilers for Triton/Inductor
if command -v /opt/rocm/llvm/bin/clang++ >/dev/null 2>&1; then
  export CC="/opt/rocm/llvm/bin/clang"
  export CXX="/opt/rocm/llvm/bin/clang++"
else
  export CC="/opt/rocm/bin/hipcc"
  export CXX="/opt/rocm/bin/hipcc"
fi

# Paths inside the container
PYUSERBASE="/workspace/pythonuserbase"
PYUSERPKG="$PYUSERBASE/lib/python3.12/site-packages"
AITER_INSTALL="/workspace/.aiter/jit/install" # Force build in /workspace

# --- FIX: Define a 100% clean PATH for the container ---
# This stops inheriting the host's /users/danizaut/.local/bin
CONTAINER_PATH="/opt/rocm/llvm/bin:/opt/rocm/bin"
CONTAINER_PATH="$CONTAINER_PATH:/opt/miniconda3/envs/pytorch/bin"
CONTAINER_PATH="$CONTAINER_PATH:/usr/local/bin:/usr/bin:/bin"
export SINGULARITYENV_PATH="$CONTAINER_PATH"
# --- END FIX ---

# Pass all required ENVs into the container
export SINGULARITYENV_CC="$CC"
export SINGULARITYENV_CXX="$CXX"
export SINGULARITYENV_HF_HOME="$HF_HOME"
export SINGULARITYENV_TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE"
export SINGULARITYENV_TORCHINDUCTOR_CACHE="$TORCHINDUCTOR_CACHE"
export SINGULARITYENV_PYTHONUSERBASE="$PYUSERBASE"
export SINGULARITYENV_PYTHONPATH="$PYUSERPKG:$AITER_INSTALL:${PYTHONPATH-}"

# Dispatcher server config
export DISPATCHER_SERVER="127.0.0.1"
export DISPATCHER_PORT="9999"
export SINGULARITYENV_DISPATCHER_SERVER="$DISPATCHER_SERVER"
export SINGULARITYENV_DISPATCHER_PORT="$DISPATCHER_PORT"

# vLLM/ROCm flags
export SINGULARITYENV_VLLM_USE_V1=1
export SINGULARITYENV_VLLM_TARGET_DEVICE=rocm
export SINGULARITYENV_VLLM_WORKER_MULTIPROC_METHOD=spawn
export SINGULARITYENV_HIP_ARCHITECTURES=gfx90a

# Bind paths as an array to avoid quote parsing errors
BINDS=(
  -B /scratch/project_462000353
  -B /flash/project_462000353
  -B /scratch/project_462000394/containers/for-turkunlp-team
  -B /pfs/lustrep3/scratch/project_462000394/containers/for-turkunlp-team
  -B /scratch/project_462000963:/scratch/project_462000963:rw
  -B "$PWD:/workspace"
)
if [ -f /usr/share/libdrm/amdgpu.ids ]; then
  BINDS+=(-B /usr/share/libdrm:/usr/share/libdrm:ro)
fi

# Helper to run inside SIF without host dotfiles
SING_EXEC() {
  singularity exec --rocm --cleanenv "${BINDS[@]}" "$IMG" bash --noprofile --norc -c "$@"
}

# Install dispatcher AND ninja
SING_EXEC "$PIP_IN_IMG install --user --upgrade 'git+https://github.com/LumiOpen/dispatcher.git' ninja"

# Cleanup trap
cleanup() {
  echo "Cleaning up server PID ${srv_pid:-}"
  kill "${srv_pid:-0}" 2>/dev/null || true
}
trap cleanup EXIT

# Start server
SING_EXEC "
  set -eux
  export PYTHONPATH=\"$PYUSERPKG:\${PYTHONPATH-}\"
  \"$PYEXEC_IN_IMG\" -m dispatcher.server \
    --infile \"$INPUT_FILE\" \
    --outfile \"$OUTPUT_FILE\" \
    --work-timeout \"$WORK_TIMEOUT\" \
    --host 0.0.0.0 \
    --port \"$DISPATCHER_PORT\"
" &
srv_pid=$!

# Wait for server
echo "Waiting for dispatcher server on 127.0.0.1:$DISPATCHER_PORT..."
for i in $(seq 1 120); do
  (echo >/dev/tcp/127.0.0.1/${DISPATCHER_PORT}) >/dev/null 2>&1 && break || true
  sleep 1
done
(echo >/dev/tcp/127.0.0.1/${DISPATCHER_PORT}) >/dev/null 2>&1 || { echo "[ERROR] dispatcher server did not start"; exit 1; }
echo "Server is up."

# Launch workers
srun -l singularity exec --rocm --cleanenv "${BINDS[@]}" "$IMG" bash --noprofile --norc -c '
  set -euxo pipefail

  # This forces os.path.expanduser("~") to return /workspace,
  # ensuring the .aiter build happens in a writable directory.
  export HOME=/workspace
  
  # Guard SLURM_LOCALID for single-task jobs
  LOCALID=${SLURM_LOCALID:-0}

  # Map GPUs
  start_gpu=$(( LOCALID * '"$GPUS_PER_TASK"' ))
  GPU_IDS=""
  for (( i=0; i<'"$GPUS_PER_TASK"'; i++ )); do
    if [ -z "$GPU_IDS" ]; then GPU_IDS="$(( start_gpu + i ))"; else GPU_IDS="${GPU_IDS},$(( start_gpu + i ))"; fi
  done
  export HIP_VISIBLE_DEVICES="$GPU_IDS"

  export MASTER_PORT=$(( 7000 + LOCALID ))
  export VLLM_PORT=$(( 8000 + LOCALID * 100 ))

  echo "Launching task $LOCALID on GPUs $HIP_VISIBLE_DEVICES"

  export PYTHONUSERBASE="/workspace/pythonuserbase"
  
  # The PATH is now the clean one from SINGULARITYENV_PATH
  export PATH="$PYTHONUSERBASE/bin:$PATH"
  
  export AITER_INSTALL="$HOME/.aiter/jit/install" # $HOME is /workspace
  export PYTHONPATH="$PYTHONUSERBASE/lib/python3.12/site-packages:$AITER_INSTALL:${PYTHONPATH-}"
  
  export HF_HOME="'"$HF_HOME"'"
  export TRANSFORMERS_CACHE="'"$TRANSFORMERS_CACHE"'"
  export TORCHINDUCTOR_CACHE="'"$TORCHINDUCTOR_CACHE"'"
  export PYTHONNOUSERSITE=
  export CC="'"$CC"'"
  export CXX="'"$CXX"'"
  export VLLM_USE_V1=1
  export VLLM_TARGET_DEVICE=rocm
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  export HIP_ARCHITECTURES=gfx90a
  export TORCH_EXTENSIONS_DIR=/dev/shm/torch_ext
  mkdir -p "$TORCH_EXTENSIONS_DIR" "$AITER_INSTALL/private_aiter/jit" 2>/dev/null || true

  # --- AIter Staging Script ---
  echo "Staging AIter module..."
  cat >/workspace/stage_aiter.py <<PY
import os, sys, glob, shutil, importlib, subprocess, pathlib

try:
    import ninja
    print(f"[stage_aiter] Ninja import OK: {ninja.__file__}")
except ImportError as e:
    print(f"[stage_aiter] FATAL: Ninja import failed, though it should be on PATH. {e!r}")
    sys.exit(1)

home=os.path.expanduser("~") # This will now be /workspace
print(f"[stage_aiter] Using HOME={home}")
jit_root=os.path.join(home,".aiter","jit")
build_root=os.path.join(jit_root,"build")
inst_root=os.path.join(home,".aiter","jit","install") # Must match PYTHONPATH
pkg_root=os.path.join(inst_root,"private_aiter")
pkg_jit=os.path.join(pkg_root,"jit")
os.makedirs(pkg_jit, exist_ok=True)
pathlib.Path(os.path.join(pkg_root,"__init__.py")).write_text("")
pathlib.Path(os.path.join(pkg_jit,"__init__.py")).write_text("")
try:
    import aiter
    from aiter.ops import enum
    print("[stage_aiter] AIter prewarm build triggered.")
except Exception as e:
    print(f"[stage_aiter] AIter prewarm raised: {e!r}")
# This is where the build process runs
hits=glob.glob(os.path.join(build_root,"**","module_aiter_enum*.so"), recursive=True)
if not hits:
    raise SystemExit("[stage_aiter] FATAL: No compiled module_aiter_enum*.so found in " + build_root)
so_src=max(hits, key=os.path.getmtime)
dst=os.path.join(pkg_jit,"module_aiter_enum.so")
if os.path.lexists(dst):
    os.remove(dst)
try:
    os.symlink(so_src,dst)
    print(f"[stage_aiter] Symlinked: {dst} -> {so_src}")
except OSError:
    shutil.copy2(so_src,dst)
    print(f"[stage_aiter] Copied: {so_src} -> {dst}")
sys.path.insert(0, inst_root)
m=importlib.import_module("private_aiter.jit.module_aiter_enum")
print("[stage_aiter] Staging complete.")
PY
  
  # Run the staging script
  "'$PYEXEC_IN_IMG'" /workspace/stage_aiter.py
  # --- End AIter Fix ---

  export MODEL="'"$MODEL"'"
  echo "Starting dispatcher task manager..."
  "'$PYEXEC_IN_IMG'" -m dispatcher.taskmanager.cli \
    --dispatcher '"$DISPATCHER_SERVER"':'"$DISPATCHER_PORT"' \
    --task '"$TASK"' \
    --batch-size '"$BATCH_SIZE"' \
    --workers '"$WORKERS"' \
    --max-model-len '"$MAX_MODEL_LEN"' \
    --tensor-parallel '"$GPUS_PER_TASK"' \
    --model '"$MODEL"' \
    --startup-timeout '"$STARTUP_TIMEOUT"' \
    --port $VLLM_PORT
'