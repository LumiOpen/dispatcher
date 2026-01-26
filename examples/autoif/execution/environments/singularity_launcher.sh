#!/bin/bash
# Singularity/Apptainer Launcher for Dispatcher SLURM jobs
#
# Usage:
#   source singularity_launcher.sh
#   run_sing_bash 'run_python -m vllm.entrypoints.openai.api_server ...'
#
# For srun workers:
#   srun -l bash -c "run_sing_bash 'your_command_here'"
#
# The launcher handles:
#   - Container bind mounts
#   - Triton cache setup (per-rank isolation)
#   - AITER package cache (JIT compiled modules + configs)
#   - ROCm/vLLM environment variables
#   - SLURM variable translation for container

###############################################################################
# Configuration Variables (override before sourcing)
###############################################################################

: "${LAUNCHER_IMG:=/shared_silo/scratch/containers/rocm_vllm_rocm7.0.0_vllm_0.11.1_20251103.sif}"
: "${LAUNCHER_PYTHON_VERSION:=3.12}"
: "${LAUNCHER_HOME:=/shared_silo/scratch/adamhrin@amd.com}"
: "${LAUNCHER_CACHE_DIR:=${LAUNCHER_HOME}/cache}"
: "${LAUNCHER_AITER_CACHE_BASE:=${LAUNCHER_HOME}/aiter_cache}"

# Export configuration so it's available in srun workers
export LAUNCHER_IMG
export LAUNCHER_PYTHON_VERSION
export LAUNCHER_HOME
export LAUNCHER_CACHE_DIR
export LAUNCHER_AITER_CACHE_BASE

# Will be set during setup and exported
export IMG=""
export LAUNCHER_AITER_DIR=""

# Python user site-packages (for pip install --user)
# Maps to /workspace/pythonuserbase inside container (workspace = $PWD)
export LAUNCHER_PYUSERBASE="${PWD}/pythonuserbase"
export LAUNCHER_PYUSERPKG="${LAUNCHER_PYUSERBASE}/lib/python${LAUNCHER_PYTHON_VERSION}/site-packages"

###############################################################################
# get_binds() - Returns bind mount arguments (one per line)
# Exported so it's available in srun workers
###############################################################################
get_binds() {
    local binds=(
        -B /shared_silo/scratch/adamhrin@amd.com:/shared_silo/scratch/adamhrin@amd.com:rw
        -B /shared_silo/scratch/models:/shared_silo/scratch/models:ro
        -B /shared_silo/scratch/datasets:/shared_silo/scratch/datasets:ro
        -B "${PWD:-$(pwd)}:/workspace"
    )
    
    # AITER package bind mount: overlay entire aiter package for writable jit/configs
    if [ -n "${LAUNCHER_AITER_DIR:-}" ]; then
        binds+=(-B "${LAUNCHER_AITER_DIR}:/usr/local/lib/python${LAUNCHER_PYTHON_VERSION}/dist-packages/aiter:rw")
    fi
    
    # libdrm for GPU info
    if [ -f /usr/share/libdrm/amdgpu.ids ]; then
        binds+=(-B /usr/share/libdrm:/usr/share/libdrm:ro)
    fi

    printf '%s\n' "${binds[@]}"
}
export -f get_binds

###############################################################################
# translate_slurm_vars() - Translate SLURM_* to APPTAINERENV_SLURM_*
# Must be called before singularity exec so SLURM vars pass through --cleanenv
###############################################################################
translate_slurm_vars() {
    local var
    for var in SLURM_PROCID SLURM_LOCALID SLURM_STEP_ID SLURM_STEP_TASK_ID \
               SLURM_JOB_ID SLURM_NODEID SLURM_NTASKS SLURM_NNODES; do
        if [ -n "${!var:-}" ]; then
            export "APPTAINERENV_${var}=${!var}"
        fi
    done
}
export -f translate_slurm_vars

###############################################################################
# setup_apptainer_environment() - Set APPTAINERENV_* variables
# These pass through --cleanenv and become regular env vars inside container
###############################################################################
setup_apptainer_environment() {
    # HuggingFace offline mode
    export APPTAINERENV_HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export APPTAINERENV_TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
    export APPTAINERENV_HF_HOME="${LAUNCHER_HOME}/hf_cache"
    
    # vLLM settings
    export APPTAINERENV_VLLM_USE_V1="${VLLM_USE_V1:-1}"
    export APPTAINERENV_VLLM_TARGET_DEVICE="rocm"
    export APPTAINERENV_VLLM_WORKER_MULTIPROC_METHOD="spawn"
    export APPTAINERENV_HIP_ARCHITECTURES="gfx942"
    
    # Optional vLLM/AITER settings (only set if defined)
    [ -n "${VLLM_USE_TRITON_FLASH_ATTN+x}" ] && export APPTAINERENV_VLLM_USE_TRITON_FLASH_ATTN="$VLLM_USE_TRITON_FLASH_ATTN"
    [ -n "${AITER_ONLINE_TUNE+x}" ] && export APPTAINERENV_AITER_ONLINE_TUNE="$AITER_ONLINE_TUNE"
    [ -n "${VLLM_ROCM_USE_AITER+x}" ] && export APPTAINERENV_VLLM_ROCM_USE_AITER="$VLLM_ROCM_USE_AITER"
    
    # Cache directories (these are base paths, actual cache dirs are set inline per-rank)
    export APPTAINERENV_XDG_CACHE_HOME="$LAUNCHER_CACHE_DIR"
    
    # Dispatcher server (if set)
    [ -n "${DISPATCHER_SERVER:-}" ] && export APPTAINERENV_DISPATCHER_SERVER="$DISPATCHER_SERVER"
    [ -n "${DISPATCHER_PORT:-}" ] && export APPTAINERENV_DISPATCHER_PORT="$DISPATCHER_PORT"
    
    # Pass through launcher config
    export APPTAINERENV_LAUNCHER_HOME="$LAUNCHER_HOME"
    export APPTAINERENV_LAUNCHER_PYTHON_VERSION="$LAUNCHER_PYTHON_VERSION"
    
    # Python user site-packages (inside container, /workspace maps to $PWD)
    export APPTAINERENV_PYTHONUSERBASE="/workspace/pythonuserbase"
}

###############################################################################
# init_aiter_cache() - Initialize aiter package cache with container's files
###############################################################################
init_aiter_cache() {
    # Use container image hash to version the cache
    local img_hash
    img_hash=$(echo "$LAUNCHER_IMG" | md5sum | cut -c1-12)
    LAUNCHER_AITER_DIR="${LAUNCHER_AITER_CACHE_BASE}/${img_hash}"
    export LAUNCHER_AITER_DIR
    mkdir -p "$LAUNCHER_AITER_DIR"
    
    # Check if cache is initialized (use __init__.py as marker)
    if [ ! -f "$LAUNCHER_AITER_DIR/__init__.py" ]; then
        echo "[launcher] Initializing aiter cache: $LAUNCHER_AITER_DIR"
        
        # Copy container's entire aiter package to our cache (WITHOUT the aiter bind mount)
        local init_binds="-B /shared_silo/scratch/adamhrin@amd.com:/shared_silo/scratch/adamhrin@amd.com:rw"
        singularity exec --rocm $init_binds "$IMG" \
            cp -r /usr/local/lib/python${LAUNCHER_PYTHON_VERSION}/dist-packages/aiter/. "$LAUNCHER_AITER_DIR/"
        
        echo "[launcher] Aiter cache initialized"
    else
        echo "[launcher] Using existing aiter cache: $LAUNCHER_AITER_DIR"
    fi
}

###############################################################################
# setup_launcher_environment() - Main setup (call once from launcher script)
###############################################################################
setup_launcher_environment() {
    # Set IMG from LAUNCHER_IMG
    IMG="$LAUNCHER_IMG"
    export IMG
    
    echo "[launcher] Using container: $IMG"
    
    # Ensure HIP_VISIBLE_DEVICES is set if SLURM set ROCR_VISIBLE_DEVICES
    if [ -n "${ROCR_VISIBLE_DEVICES:-}" ] && [ -z "${HIP_VISIBLE_DEVICES:-}" ]; then
        export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
    fi
    
    # Create directories
    mkdir -p logs
    mkdir -p "$LAUNCHER_CACHE_DIR"
    mkdir -p "${LAUNCHER_HOME}/hf_cache"
    mkdir -p "$LAUNCHER_PYUSERBASE"
    
    # Initialize aiter cache (copies entire aiter package if needed)
    init_aiter_cache
    
    # Set up APPTAINERENV_* variables
    setup_apptainer_environment
    
    # Install ninja (required for AITER JIT compilation)
    echo "[launcher] Installing ninja..."
    local binds_array
    mapfile -t binds_array < <(get_binds)
    
    singularity exec --rocm --cleanenv "${binds_array[@]}" "$IMG" \
        bash --noprofile --norc -c "
            export HOME=${LAUNCHER_HOME}
            export PYTHONUSERBASE=/workspace/pythonuserbase
            export PATH=\$PYTHONUSERBASE/bin:\$PATH
            python3 -m pip install --user --upgrade ninja
        "
    
    # Pre-initialize AITER before any workers start
    echo "[launcher] Pre-initializing AITER..."
    singularity exec --rocm --cleanenv "${binds_array[@]}" "$IMG" \
        bash --noprofile --norc -c "export HOME=${LAUNCHER_HOME}; python3 -c 'import aiter; print(\"AITER initialized successfully\")'"
    
    echo "[launcher] Setup complete"
}

###############################################################################
# run_sing_bash "command" - Run bash commands inside container
# This function is exported and works inside srun workers
###############################################################################
run_sing_bash() {
    # Check that IMG is set (indicates setup was done)
    [ -n "${IMG:-}" ] || {
        echo "[launcher] ERROR: run_sing_bash called before setup_launcher_environment" >&2
        return 1
    }
    
    if [ $# -eq 0 ]; then
        echo "[launcher] ERROR: no command provided" >&2
        return 1
    fi
    
    # Translate SLURM variables for this worker (srun sets different values per rank)
    translate_slurm_vars
    
    # Get bind mounts fresh (includes LAUNCHER_AITER_DIR)
    local binds_array
    mapfile -t binds_array < <(get_binds)
    
    # Build inline environment setup that runs inside the container
    # This handles per-rank cache isolation and defines run_python helper
    local env_setup="
    # Set HOME
    export HOME=\"${LAUNCHER_HOME}\"
    
    # Python user site-packages
    export PYTHONUSERBASE=\"\${PYTHONUSERBASE:-/workspace/pythonuserbase}\"
    export PATH=\"\$PYTHONUSERBASE/bin:\$PATH\"
    export PYTHONPATH=\"\$PYTHONUSERBASE/lib/python${LAUNCHER_PYTHON_VERSION}/site-packages:/usr/local/lib/python${LAUNCHER_PYTHON_VERSION}/dist-packages:\${PYTHONPATH:-}\"
    
    # Per-rank Triton cache isolation (avoids multi-rank races on shared filesystems)
    export TRITON_CACHE_DIR=\"/tmp/triton_cache/\${SLURM_JOB_ID:-nojob}/\${SLURM_PROCID:-\${SLURM_LOCALID:-0}}\"
    export XDG_CACHE_HOME=\"/tmp/xdg_cache/\${SLURM_JOB_ID:-nojob}/\${SLURM_PROCID:-\${SLURM_LOCALID:-0}}\"
    mkdir -p \"\$TRITON_CACHE_DIR\" \"\$XDG_CACHE_HOME\" 2>/dev/null || true
    
    # Torch extensions in shared memory
    export TORCH_EXTENSIONS_DIR=\"\${TORCH_EXTENSIONS_DIR:-/dev/shm/torch_ext}\"
    mkdir -p \"\$TORCH_EXTENSIONS_DIR\" 2>/dev/null || true
    
    # Define run_python helper
    run_python() {
        python3 \"\$@\"
    }
"
    
    # Combine setup with user command
    local full_command="${env_setup}
${*}"
    
    # Execute with --cleanenv (only APPTAINERENV_* vars pass through)
    singularity exec --rocm --cleanenv "${binds_array[@]}" "$IMG" \
        bash --noprofile --norc -c "$full_command"
}
export -f run_sing_bash

###############################################################################
# run_sing_python [args] - Run Python directly in container
# Sets up PYTHONUSERBASE and PYTHONPATH for user site-packages
###############################################################################
run_sing_python() {
    [ -n "${IMG:-}" ] || {
        echo "[launcher] ERROR: run_sing_python called before setup_launcher_environment" >&2
        return 1
    }
    
    local binds_array
    mapfile -t binds_array < <(get_binds)
    
    # Build full command with environment setup and python call
    local full_cmd="
export HOME=\"${LAUNCHER_HOME}\"
export PYTHONUSERBASE=\"/workspace/pythonuserbase\"
export PATH=\"\$PYTHONUSERBASE/bin:\$PATH\"
export PYTHONPATH=\"\$PYTHONUSERBASE/lib/python${LAUNCHER_PYTHON_VERSION}/site-packages:/usr/local/lib/python${LAUNCHER_PYTHON_VERSION}/dist-packages:\${PYTHONPATH:-}\"
python3 \"\$@\"
"
    
    singularity exec --rocm --cleanenv "${binds_array[@]}" "$IMG" \
        bash --noprofile --norc -c "$full_cmd" bash "$@"
}
export -f run_sing_python

###############################################################################
# Cleanup trap
###############################################################################
_cleanup() {
    kill "${srv_pid:-0}" 2>/dev/null || true
}
trap _cleanup EXIT

###############################################################################
# Auto-setup when sourced
###############################################################################
setup_launcher_environment
