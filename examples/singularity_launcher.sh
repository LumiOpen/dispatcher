#!/bin/bash
# Launcher library for Dispatcher SLURM jobs with Singularity containers
# Source this file in your SLURM job scripts to avoid duplicating container setup code

###############################################################################
# Configuration Variables (can be overridden before sourcing this file)
###############################################################################

# Default container and cache paths
: "${LAUNCHER_IMG:=/shared_silo/scratch/containers/rocm_vllm_rocm7.0.0_vllm_0.11.1_20251103.sif}"
: "${LAUNCHER_PYEXEC_IN_IMG:=python3}"
: "${LAUNCHER_PYTHON_VERSION:=3.12}"

# Offline mode flags (set to empty string to disable)
: "${LAUNCHER_HF_HUB_OFFLINE:=1}"
: "${LAUNCHER_TRANSFORMERS_OFFLINE:=1}"

###############################################################################
# setup_singularity_environment()
# Sets up all environment variables and paths for Singularity container execution
###############################################################################
setup_singularity_environment() {
  # Create necessary directories
  mkdir -p logs pythonuserbase

  # Caches MUST be on a writable, bound path
  export HF_HOME="/shared_silo/scratch/adamhrin@amd.com/hf_cache"
  export TRANSFORMERS_CACHE="$HF_HOME"
  export TORCHINDUCTOR_CACHE="/shared_silo/scratch/adamhrin@amd.com/torch_inductor_cache"
  export TRITON_DISABLE_CACHE=1
  mkdir -p "$HF_HOME" "$TORCHINDUCTOR_CACHE"

  # Set offline mode flags if configured
  if [ -n "$LAUNCHER_HF_HUB_OFFLINE" ]; then
    export HF_HUB_OFFLINE="$LAUNCHER_HF_HUB_OFFLINE"
  fi
  if [ -n "$LAUNCHER_TRANSFORMERS_OFFLINE" ]; then
    export TRANSFORMERS_OFFLINE="$LAUNCHER_TRANSFORMERS_OFFLINE"
  fi

  # Container and Python paths
  export IMG="$LAUNCHER_IMG"
  export PYEXEC_IN_IMG="$LAUNCHER_PYEXEC_IN_IMG"
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
  export PYUSERBASE="/workspace/pythonuserbase"
  export PYUSERPKG="$PYUSERBASE/lib/python${LAUNCHER_PYTHON_VERSION}/site-packages"

  # --- Define a 100% clean PATH for the container ---
  # This stops inheriting host paths that can cause issues
  CONTAINER_PATH="/opt/rocm/llvm/bin:/opt/rocm/bin"
  CONTAINER_PATH="$CONTAINER_PATH:/opt/miniconda3/envs/pytorch/bin"
  CONTAINER_PATH="$CONTAINER_PATH:/usr/local/bin:/usr/bin:/bin"
  export SINGULARITYENV_PATH="$CONTAINER_PATH"

  # Pass all required ENVs into the container
  export SINGULARITYENV_CC="$CC"
  export SINGULARITYENV_CXX="$CXX"
  export SINGULARITYENV_HF_HOME="$HF_HOME"
  export SINGULARITYENV_TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE"
  export SINGULARITYENV_TORCHINDUCTOR_CACHE="$TORCHINDUCTOR_CACHE"
  export SINGULARITYENV_TRITON_DISABLE_CACHE=1
  export SINGULARITYENV_PYTHONUSERBASE="$PYUSERBASE"
  export SINGULARITYENV_PYTHONPATH="$PYUSERPKG:\${PYTHONPATH-}"
  export SINGULARITYENV_PYEXEC_IN_IMG="$PYEXEC_IN_IMG"
  export SINGULARITYENV_DISPATCHER_SERVER="${DISPATCHER_SERVER}"
  export SINGULARITYENV_DISPATCHER_PORT="${DISPATCHER_PORT}"
  
  if [ -n "${HF_HUB_OFFLINE:-}" ]; then
    export SINGULARITYENV_HF_HUB_OFFLINE="$HF_HUB_OFFLINE"
  fi
  if [ -n "${TRANSFORMERS_OFFLINE:-}" ]; then
    export SINGULARITYENV_TRANSFORMERS_OFFLINE="$TRANSFORMERS_OFFLINE"
  fi

  # vLLM/ROCm flags
  export SINGULARITYENV_SAFETENSORS_FAST_GPU=1
  export SINGULARITYENV_VLLM_ROCM_USE_AITER=1
  export SINGULARITYENV_VLLM_USE_V1=1
  export SINGULARITYENV_VLLM_TARGET_DEVICE=rocm
  export SINGULARITYENV_VLLM_WORKER_MULTIPROC_METHOD=spawn
  export SINGULARITYENV_HIP_ARCHITECTURES=gfx942
  
  # Worker environment variables (for use inside container)
  export SINGULARITYENV_TORCH_EXTENSIONS_DIR=/dev/shm/torch_ext
  export SINGULARITYENV_PYTHONNOUSERSITE=
}

###############################################################################
# get_binds
# Returns bind mount arguments as an array
# This function-based approach ensures BINDS are always available regardless of sourcing context
###############################################################################
get_binds() {
  local binds=(
    -B /shared_silo/scratch/adamhrin@amd.com:/shared_silo/scratch/adamhrin@amd.com:rw
    -B /shared_silo/scratch/models:/shared_silo/scratch/models:ro
    -B "${PWD:-$(pwd)}:/workspace"
  )
  if [ -f /usr/share/libdrm/amdgpu.ids ]; then
    binds+=(-B /usr/share/libdrm:/usr/share/libdrm:ro)
  fi
  printf '%s\n' "${binds[@]}"
}
# Export function so it's available in srun workers
export -f get_binds

###############################################################################
# install_dispatcher_packages
# Installs dispatcher and ninja in the container
###############################################################################
install_dispatcher_packages() {
  local binds_array
  mapfile -t binds_array < <(get_binds)
  singularity exec --rocm --cleanenv "${binds_array[@]}" "$IMG" bash --noprofile --norc -c \
    "$PIP_IN_IMG install --user --upgrade -e /shared_silo/scratch/adamhrin@amd.com/dispatcher ninja"
}

###############################################################################
# import_container_config
# Imports container configuration from parent environment into worker context
# Variables are already available via SINGULARITYENV_* from host, just ensure they're exported
###############################################################################
import_container_config() {
  # These variables are passed via SINGULARITYENV_* and available in container
  # Just ensure they're exported (they should already be set by Singularity)
  export HF_HOME="${HF_HOME:-}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-}"
  export TORCHINDUCTOR_CACHE="${TORCHINDUCTOR_CACHE:-}"
  export CC="${CC:-}"
  export CXX="${CXX:-}"
  export PYEXEC_IN_IMG="${PYEXEC_IN_IMG:-}"
}

###############################################################################
# run_aiter_staging
# Runs the AIter staging script inside a container
###############################################################################
run_aiter_staging() {
  get_aiter_staging_script > /workspace/stage_aiter.py
  run_python /workspace/stage_aiter.py
}

###############################################################################
# run_python
# Helper function to run Python commands when inside a container
# Usage: run_python -m module.name args...  or  run_python script.py args...
###############################################################################
run_python() {
  is_inside_container || {
    echo "[singularity_launcher] ERROR: run_python called outside container" >&2
    return 1
  }
  "${PYEXEC_IN_IMG:-python3}" "$@"
}

###############################################################################
# translate_slurm_vars
# Translates SLURM_* environment variables to SINGULARITYENV_* versions
# so they pass through --cleanenv
###############################################################################
translate_slurm_vars() {
  local var
  for var in SLURM_PROCID SLURM_LOCALID SLURM_STEP_ID SLURM_STEP_TASK_ID SLURM_JOB_ID SLURM_NODEID SLURM_NTASKS; do
    if [ -n "${!var:-}" ]; then
      export "SINGULARITYENV_${var}=${!var}"
    fi
  done
}
# Export function so it's available in srun workers
export -f translate_slurm_vars

###############################################################################
# Complete environment setup
###############################################################################
setup_launcher_environment() {
  translate_slurm_vars
  setup_singularity_environment
  install_dispatcher_packages
}

###############################################################################
# run_sing_bash
# Helper function to run bash commands inside the Singularity container
# Automatically translates SLURM_* variables to SINGULARITYENV_* before execution
# Automatically sets up worker environment inline (no external script needed)
# Usage: run_sing_bash "command to run"
###############################################################################
run_sing_bash() {
  [ -n "${IMG:-}" ] || {
    echo "[singularity_launcher] ERROR: run_sing_bash called before setup_singularity_environment" >&2
    return 1
  }
  translate_slurm_vars
  # Use get_binds() function to get bind mounts - works regardless of sourcing context
  local binds_array
  mapfile -t binds_array < <(get_binds)
  # Build inline environment setup command
  # This sets up the worker environment directly without needing an external script
  local env_setup="
    # Set HOME to /workspace
    export HOME=/workspace
    
    # Import container configuration from SINGULARITYENV_* variables
    export HF_HOME=\"\${HF_HOME:-}\"
    export TRANSFORMERS_CACHE=\"\${TRANSFORMERS_CACHE:-}\"
    export TORCHINDUCTOR_CACHE=\"\${TORCHINDUCTOR_CACHE:-}\"
    export HF_HUB_OFFLINE=\"\${HF_HUB_OFFLINE:-}\"
    export TRANSFORMERS_OFFLINE=\"\${TRANSFORMERS_OFFLINE:-}\"
    export CC=\"\${CC:-}\"
    export CXX=\"\${CXX:-}\"
    export PYEXEC_IN_IMG=\"\${PYEXEC_IN_IMG:-}\"
    export TRITON_DISABLE_CACHE=\"\${TRITON_DISABLE_CACHE:-1}\"
    
    # Setup Python environment
    export PYTHONUSERBASE=\"/workspace/pythonuserbase\"
    export PATH=\"\$PYTHONUSERBASE/bin:\$PATH\"
    export PYTHONPATH=\"\$PYTHONUSERBASE/lib/python${LAUNCHER_PYTHON_VERSION}/site-packages:\${PYTHONPATH-}\"
    export PYTHONNOUSERSITE=
    
    # vLLM/ROCm flags
    export SAFETENSORS_FAST_GPU=\${SAFETENSORS_FAST_GPU:-1}
    export VLLM_ROCM_USE_AITER=\${VLLM_ROCM_USE_AITER:-1}
    export VLLM_USE_V1=\${VLLM_USE_V1:-1}
    export VLLM_TARGET_DEVICE=\${VLLM_TARGET_DEVICE:-rocm}
    export VLLM_WORKER_MULTIPROC_METHOD=\${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
    export HIP_ARCHITECTURES=\${HIP_ARCHITECTURES:-gfx942}
    
    # Create necessary directories
    export TORCH_EXTENSIONS_DIR=/dev/shm/torch_ext
    mkdir -p \"\$TORCH_EXTENSIONS_DIR\" 2>/dev/null || true
    
    # Define run_python helper function
    run_python() {
      \"\${PYEXEC_IN_IMG:-python3}\" \"\$@\"
    }
  "
  # Execute command with inline environment setup
  # Ensure we have at least one argument
  if [ $# -eq 0 ]; then
    echo "[singularity_launcher] ERROR: run_sing_bash called without command" >&2
    return 1
  fi
  # Join all arguments with spaces (typical case: single multi-line command string)
  local user_command="$*"
  local full_command="$env_setup
$user_command"
  singularity exec --rocm --cleanenv "${binds_array[@]}" "$IMG" bash --noprofile --norc -c "$full_command"
}
# Export function so it's available in srun workers without needing to source the launcher
export -f run_sing_bash

###############################################################################
# run_sing_python
# Helper function to run Python commands inside the Singularity container
# PYTHONPATH is already configured via SINGULARITYENV_PYTHONPATH from setup_singularity_environment
# Usage: run_sing_python -m module.name --arg1 val1 --arg2 val2
###############################################################################
run_sing_python() {
  [ -n "${IMG:-}" ] && [ -n "${PYEXEC_IN_IMG:-}" ] || {
    echo "[singularity_launcher] ERROR: run_sing_python called before setup_singularity_environment" >&2
    return 1
  }
  # Use get_binds() function to get bind mounts - works regardless of sourcing context
  local binds_array
  mapfile -t binds_array < <(get_binds)
  singularity exec --rocm --cleanenv "${binds_array[@]}" "$IMG" "$PYEXEC_IN_IMG" "$@"
}

###############################################################################
# is_inside_container
# Detects if we're running inside a Singularity container
# Returns 0 if inside container, 1 if not
###############################################################################
is_inside_container() {
  [ -n "${SINGULARITY_NAME:-}" ] || [ -f /.singularity.d/env/99-base.sh ]
}

# Run this when the script is sourced in the launcher
setup_launcher_environment
