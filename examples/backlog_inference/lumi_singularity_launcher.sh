#!/bin/bash
# Launcher library for SLURM jobs with Singularity containers on LUMI
# Source this file in your SLURM job scripts to avoid duplicating container setup code
#
# LUMI equivalent of tw_singularity_launcher.sh. Differences from the TW version:
#   - loads the LUMI AI Framework (LAIFS) module + lumi-multitorch container
#     instead of the TW rocm_vllm/vllm-openai-rocm sif
#   - binds LUMI's Lustre tiers (/scratch,/project,/flash,/pfs,...) 1:1 (no
#     path remapping to /workspace) instead of TW's /shared_silo/scratch paths
#   - no --rocm flag on singularity exec / no AITER JIT patching: LUMI's
#     lumi-aif-singularity-bindings module handles ROCm device/library binds,
#     and AITER's JIT install-root hack in the TW launcher is a workaround
#     specific to MI300 (gfx942); LUMI is MI250x (gfx90a) and doesn't need it
#   - no hardcoded SINGULARITYENV_PATH: the container's own PATH is used as-is

###############################################################################
# Configuration Variables (can be overridden before sourcing this file)
###############################################################################

: "${LAUNCHER_IMG:=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-plus-u24r70f21m50t210-20260807_115122.sif}"
: "${LAUNCHER_PYEXEC_IN_IMG:=python3}"
: "${LAUNCHER_PYTHON_VERSION:=3.12}"
: "${LAUNCHER_HF_HOME:=/scratch/project_462001516/users/zosaelai2/hf-home}"
: "${LAUNCHER_TORCHINDUCTOR_CACHE:=/tmp/$USER/$SLURM_JOB_ID/torch_inductor_cache}"
: "${LAUNCHER_BIND_DIRS:=/pfs,/scratch,/projappl,/project,/flash,/appl,/opt/cray,/var/spool/slurmd}"
: "${LAUNCHER_PASSTHROUGH_VARS:=}"

# Offline mode flags (set to empty string to disable)
: "${LAUNCHER_HF_HUB_OFFLINE:=}"
: "${LAUNCHER_TRANSFORMERS_OFFLINE:=}"

###############################################################################
# setup_singularity_environment()
# Loads the LAIFS module (which wires up ROCm device/library binds for
# singularity) and sets up all environment variables and paths for container
# execution.
###############################################################################
setup_singularity_environment() {
  module purge
  module use /appl/local/laifs/modules
  module load lumi-aif-singularity-bindings

  mkdir -p logs pythonuserbase

  # Caches MUST be on writable, node-local paths. Scope them by user and
  # Slurm job so stale cache parents from other users cannot block mkdir.
  export HF_HOME="$LAUNCHER_HF_HOME"
  export TRANSFORMERS_CACHE="$HF_HOME"
  export TORCHINDUCTOR_CACHE="$LAUNCHER_TORCHINDUCTOR_CACHE"
  install -d -m 700 "/tmp/$USER/$SLURM_JOB_ID" "/dev/shm/$USER/$SLURM_JOB_ID"
  mkdir -p "$HF_HOME" "$TORCHINDUCTOR_CACHE"

  if [ -n "$LAUNCHER_HF_HUB_OFFLINE" ]; then
    export HF_HUB_OFFLINE="$LAUNCHER_HF_HUB_OFFLINE"
  fi
  if [ -n "$LAUNCHER_TRANSFORMERS_OFFLINE" ]; then
    export TRANSFORMERS_OFFLINE="$LAUNCHER_TRANSFORMERS_OFFLINE"
  fi

  export IMG="$LAUNCHER_IMG"
  export PYEXEC_IN_IMG="$LAUNCHER_PYEXEC_IN_IMG"
  export PIP_IN_IMG="$PYEXEC_IN_IMG -m pip"

  export PYUSERBASE="$PWD/pythonuserbase"
  export PYUSERPKG="$PYUSERBASE/lib/python${LAUNCHER_PYTHON_VERSION}/site-packages"

  # HF token: read from env, then fall back to the default cached token file
  if [ -z "${HF_TOKEN:-}" ]; then
    local _token_file="$HOME/.cache/huggingface/token"
    if [ -f "$_token_file" ]; then
      HF_TOKEN="$(cat "$_token_file")"
    fi
  fi
  if [ -n "${HF_TOKEN:-}" ]; then
    { set +x; } 2>/dev/null
    export SINGULARITYENV_HF_TOKEN="$HF_TOKEN"
    set -x
  fi

  # Pass all required ENVs into the container
  export SINGULARITYENV_USER="$USER"
  export SINGULARITYENV_HF_HOME="$HF_HOME"
  export SINGULARITYENV_TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE"
  export SINGULARITYENV_TORCHINDUCTOR_CACHE="$TORCHINDUCTOR_CACHE"
  export SINGULARITYENV_PYTHONUSERBASE="$PYUSERBASE"
  export SINGULARITYENV_PYTHONPATH="$PYUSERPKG:\${PYTHONPATH-}"
  export SINGULARITYENV_PYEXEC_IN_IMG="$PYEXEC_IN_IMG"
  export SINGULARITYENV_DISPATCHER_SERVER="${DISPATCHER_SERVER:-}"

  # Keep --cleanenv while allowing each job to explicitly opt selected host
  # variables into the container environment.
  local passthrough_var
  for passthrough_var in $LAUNCHER_PASSTHROUGH_VARS; do
    if [[ ! "$passthrough_var" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
      echo "Invalid LAUNCHER_PASSTHROUGH_VARS entry: $passthrough_var" >&2
      return 1
    fi
    if declare -p "$passthrough_var" &>/dev/null; then
      export "SINGULARITYENV_${passthrough_var}=${!passthrough_var}"
    fi
  done

  if [ -n "${HF_HUB_OFFLINE:-}" ]; then
    export SINGULARITYENV_HF_HUB_OFFLINE="$HF_HUB_OFFLINE"
  fi
  if [ -n "${TRANSFORMERS_OFFLINE:-}" ]; then
    export SINGULARITYENV_TRANSFORMERS_OFFLINE="$TRANSFORMERS_OFFLINE"
  fi

  # vLLM/ROCm flags (LUMI is gfx90a / MI250x -- no AITER, no gfx942 pin)
  export SINGULARITYENV_VLLM_USE_V1=${VLLM_USE_V1:-1}
  export SINGULARITYENV_VLLM_TARGET_DEVICE=rocm
  export SINGULARITYENV_VLLM_WORKER_MULTIPROC_METHOD=spawn

  export SINGULARITYENV_TORCH_EXTENSIONS_DIR="/dev/shm/$USER/$SLURM_JOB_ID/torch_ext"
  export SINGULARITYENV_PYTHONNOUSERSITE=
}

###############################################################################
# get_binds
# Returns bind mount arguments as an array. LUMI containers expect host paths
# bound 1:1 (no /workspace remap) -- matches the pattern used by
# train-gpt-prelude-iter0830400-128k-packed.sh.
###############################################################################
get_binds() {
  local binds=(
    -B "${PWD:-$(pwd)}"
    -B "$LAUNCHER_BIND_DIRS"
  )
  printf '%s\n' "${binds[@]}"
}
export -f get_binds

###############################################################################
# setup_cleanup_trap
###############################################################################
setup_cleanup_trap() {
  trap 'kill "${srv_pid:-0}" 2>/dev/null || true' EXIT
}

###############################################################################
# translate_slurm_vars
# Translates SLURM_* environment variables to SINGULARITYENV_* versions
# so they pass through --cleanenv
###############################################################################
translate_slurm_vars() {
  local var
  for var in SLURM_PROCID SLURM_LOCALID SLURM_STEP_ID SLURM_STEP_TASK_ID SLURM_JOB_ID SLURM_NODEID SLURM_NTASKS SLURM_JOB_GPUS; do
    if [ -n "${!var:-}" ]; then
      export "SINGULARITYENV_${var}=${!var}"
    fi
  done
}
export -f translate_slurm_vars

###############################################################################
# Complete environment setup
###############################################################################
setup_launcher_environment() {
  translate_slurm_vars
  setup_singularity_environment
  setup_cleanup_trap
}

###############################################################################
# run_sing_bash
# Helper function to run bash commands inside the Singularity container
# Usage: run_sing_bash "command to run"
###############################################################################
run_sing_bash() {
  [ -n "${IMG:-}" ] || {
    echo "[lumi_singularity_launcher] ERROR: run_sing_bash called before setup_singularity_environment" >&2
    return 1
  }
  translate_slurm_vars
  local binds_array
  mapfile -t binds_array < <(get_binds)
  local env_setup="
    export HOME=\"$PWD\"

    export HF_HOME=\"\${HF_HOME:-}\"
    export TRANSFORMERS_CACHE=\"\${TRANSFORMERS_CACHE:-}\"
    export TORCHINDUCTOR_CACHE=\"\${TORCHINDUCTOR_CACHE:-}\"
    export HF_HUB_OFFLINE=\"\${HF_HUB_OFFLINE:-}\"
    export TRANSFORMERS_OFFLINE=\"\${TRANSFORMERS_OFFLINE:-}\"
    export PYEXEC_IN_IMG=\"\${PYEXEC_IN_IMG:-}\"

    export PYTHONUSERBASE=\"\${PYTHONUSERBASE:-$PWD/pythonuserbase}\"
    export PATH=\"\$PYTHONUSERBASE/bin:\$PATH\"
    export PYTHONPATH=\"\$PYTHONUSERBASE/lib/python${LAUNCHER_PYTHON_VERSION}/site-packages:\${PYTHONPATH-}\"
    export PYTHONNOUSERSITE=

    export VLLM_USE_V1=\${VLLM_USE_V1:-1}
    export VLLM_TARGET_DEVICE=\${VLLM_TARGET_DEVICE:-rocm}
    export VLLM_WORKER_MULTIPROC_METHOD=\${VLLM_WORKER_MULTIPROC_METHOD:-spawn}

    # Triton cache isolation (avoid multi-rank races on shared filesystems)
    export TRITON_CACHE_DIR=\"/tmp/\$USER/\$SLURM_JOB_ID/triton_cache/\${SLURM_PROCID:-\${SLURM_LOCALID:-0}}\"
    export XDG_CACHE_HOME=\"/tmp/\$USER/\$SLURM_JOB_ID/xdg_cache/\${SLURM_PROCID:-\${SLURM_LOCALID:-0}}\"
    mkdir -p \"\$TRITON_CACHE_DIR\" \"\$XDG_CACHE_HOME\" 2>/dev/null || true

    export TORCH_EXTENSIONS_DIR=\"\${TORCH_EXTENSIONS_DIR:-/dev/shm/\$USER/\$SLURM_JOB_ID/torch_ext}\"
    mkdir -p \"\$TORCH_EXTENSIONS_DIR\" 2>/dev/null || true

    run_python() {
      \"\${PYEXEC_IN_IMG:-python3}\" \"\$@\"
    }
  "
  if [ $# -eq 0 ]; then
    echo "[lumi_singularity_launcher] ERROR: run_sing_bash called without command" >&2
    return 1
  fi
  local user_command="$*"
  local full_command="$env_setup
$user_command"
  singularity exec --cleanenv "${binds_array[@]}" "$IMG" bash --noprofile --norc -c "$full_command"
}
export -f run_sing_bash

###############################################################################
# run_sing_python
###############################################################################
run_sing_python() {
  [ -n "${IMG:-}" ] && [ -n "${PYEXEC_IN_IMG:-}" ] || {
    echo "[lumi_singularity_launcher] ERROR: run_sing_python called before setup_singularity_environment" >&2
    return 1
  }
  local binds_array
  mapfile -t binds_array < <(get_binds)
  singularity exec --cleanenv "${binds_array[@]}" "$IMG" "$PYEXEC_IN_IMG" "$@"
}

###############################################################################
# is_inside_container
###############################################################################
is_inside_container() {
  [ -n "${SINGULARITY_NAME:-}" ] || [ -f /.singularity.d/env/99-base.sh ]
}

# Run this when the script is sourced in the launcher
setup_launcher_environment
