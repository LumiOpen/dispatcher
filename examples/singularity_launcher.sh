#!/bin/bash
# Launcher library for Dispatcher SLURM jobs with Singularity containers
# Source this file in your SLURM job scripts to avoid duplicating container setup code

###############################################################################
# Configuration Variables (can be overridden before sourcing this file)
###############################################################################

# Default container and cache paths
: "${LAUNCHER_IMG:=/shared_silo/scratch/containers/7.x-preview_rocm7.2_preview_ubuntu_22.04_vlm_0.10.1_instinct_20251029.sif}"
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
  # based on https://rocm.docs.amd.com/en/docs-7.0-docker/benchmark-docker/inference-vllm-deepseek-r1-fp8.html
  # Note: this flag VLLM_ROCM_QUICK_REDUCE_QUANTIZATION may not be compatible with MI325X GPUs
  # export SINGULARITYENV_VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
  export SINGULARITYENV_VLLM_TARGET_DEVICE=rocm
  export SINGULARITYENV_VLLM_WORKER_MULTIPROC_METHOD=spawn
  export SINGULARITYENV_HIP_ARCHITECTURES=gfx942
  # Work around missing AIter fused GEMM config JSONs in some container builds.
  # Disables the Triton fused shared-experts path that tries to open
  # `.../aiter/ops/triton/configs/gemm/MI300X-FUSED-GEMM-A8W8_BLOCKSCALE-A16W16.json`.
  export SINGULARITYENV_VLLM_ROCM_USE_AITER_TRITON_FUSED_SHARED_EXPERTS=0
  
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
# setup_cleanup_trap
# Sets up cleanup trap to kill server process on exit
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
  setup_cleanup_trap
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
    
    # Setup Python environment
    export PYTHONUSERBASE=\"\${PYTHONUSERBASE:-/workspace/pythonuserbase}\"
    export PATH=\"\$PYTHONUSERBASE/bin:\$PATH\"
    # AIter JIT writes importable artifacts into a writable install root.
    # Without this, it may attempt to write into system site-packages and later fail imports like:
    #   No module named 'aiter.jit.module_gemm_common'
    export AITER_INSTALL=\"\$HOME/.aiter/jit/install\"
    mkdir -p \"\$AITER_INSTALL\" \"\$AITER_INSTALL/aiter/jit\" \"\$AITER_INSTALL/private_aiter/jit\" 2>/dev/null || true
    export PYTHONPATH=\"\$PYTHONUSERBASE/lib/python${LAUNCHER_PYTHON_VERSION}/site-packages:\$AITER_INSTALL:\${PYTHONPATH-}\"
    export PYTHONNOUSERSITE=
    # Enable AIter import diagnostics by default (can be disabled by setting AITER_IMPORT_DEBUG=0).
    export AITER_IMPORT_DEBUG=\"\${AITER_IMPORT_DEBUG:-1}\"

    # Ensure Python can import AIter-built extension modules (e.g. module_gemm_common.so).
    # Root issue: `aiter.jit.core` builds modules under `~/.aiter/jit/` but imports them as
    # `aiter.jit.<module>` from the installed package path (dist-packages/aiter/jit), which
    # doesn't include `~/.aiter/jit` on `aiter.jit.__path__`.
    #
    # Using a `.pth` file in the *user site-packages* is the most reliable way to run early,
    # for every Python process (including vLLM multiprocess workers).
    user_site=\"\$PYTHONUSERBASE/lib/python${LAUNCHER_PYTHON_VERSION}/site-packages\"
    mkdir -p \"\$user_site\" 2>/dev/null || true
    cat >\"\$user_site/aiter_jit_pathfix.pth\" <<'PTH'
import os,sys,importlib; aj=importlib.import_module('aiter.jit'); p=os.path.join(os.path.expanduser('~'),'.aiter','jit'); (aj.__path__.append(p) if (os.path.isdir(p) and p not in list(getattr(aj,'__path__',[]))) else None); dbg=os.environ.get('AITER_IMPORT_DEBUG','0'); (sys.stderr.write('[aiter_import_debug] aiter.jit.__path__ extended with '+p+'\\n') if dbg not in ('0','','false','False','no','No') else None)
PTH

    # Make AIter's build output importable as `aiter.jit.*`.
    # AIter often writes compiled extension modules (e.g. module_gemm_common.so) under:
    #   \$HOME/.aiter/jit/
    # but the Python package `aiter.jit` lives under system site-packages.
    # We bridge that gap by extending `aiter.jit.__path__` at interpreter startup.
    cat >\"\$AITER_INSTALL/sitecustomize.py\" <<'PY'
import os
import sys
import glob
import importlib
import importlib.util

def _dbg(msg: str) -> None:
    # NOTE: avoid double quotes in this file because it's embedded inside a bash
    # double-quoted string in the launcher.
    if os.environ.get('AITER_IMPORT_DEBUG', '0') not in ('0', '', 'false', 'False', 'no', 'No'):
        try:
            sys.stderr.write(f'[aiter_import_debug] {msg}\n')
            sys.stderr.flush()
        except Exception:
            pass

def _maybe_extend_aiter_jit_path() -> None:
    _dbg(f'sitecustomize loaded: sys.executable={sys.executable}')
    _dbg('env HOME={} AITER_INSTALL={} PYTHONPATH={}'.format(
        os.environ.get('HOME'),
        os.environ.get('AITER_INSTALL'),
        os.environ.get('PYTHONPATH'),
    ))
    _dbg('env SLURM_PROCID={} SLURM_LOCALID={} SLURM_JOB_ID={}'.format(
        os.environ.get('SLURM_PROCID'),
        os.environ.get('SLURM_LOCALID'),
        os.environ.get('SLURM_JOB_ID'),
    ))
    try:
        _dbg(f'sys.path[0:8]={sys.path[0:8]}')
    except Exception:
        pass
    try:
        import aiter
        _dbg('aiter imported from: {}'.format(getattr(aiter, '__file__', None)))
        _dbg('aiter version: {}'.format(getattr(aiter, '__version__', None)))
    except Exception:
        _dbg('import aiter failed (skipping aiter.jit path extension)')
        return

    try:
        import aiter.jit as aiter_jit
        extra = os.path.join(os.path.expanduser('~'), '.aiter', 'jit')
        if os.path.isdir(extra) and extra not in list(getattr(aiter_jit, '__path__', [])):
            aiter_jit.__path__.append(extra)  # type: ignore[attr-defined]
            _dbg(f'extended aiter.jit.__path__ with: {extra}')
        _dbg('aiter.jit imported from: {}'.format(getattr(aiter_jit, '__file__', None)))
        _dbg('aiter.jit.__path__={}'.format(list(getattr(aiter_jit, '__path__', []))))

        target = 'aiter.jit.module_gemm_common'
        try:
            spec = importlib.util.find_spec(target)
            _dbg(f'find_spec({target}) -> {spec}')
        except Exception as e:
            _dbg(f'find_spec({target}) raised: {e!r}')

        try:
            for p in list(getattr(aiter_jit, '__path__', [])):
                hits = sorted(glob.glob(os.path.join(p, 'module_gemm_common*')))
                if hits:
                    _dbg(f'candidate files in {p}: {hits[:20]}')
        except Exception as e:
            _dbg(f'globbing for module_gemm_common artifacts raised: {e!r}')

        try:
            m = importlib.import_module(target)
            _dbg('imported {} from: {}'.format(target, getattr(m, '__file__', None)))
        except Exception as e:
            _dbg(f'import {target} failed: {e!r}')
    except Exception:
        _dbg('unexpected exception while extending aiter.jit.__path__ (suppressed)')
        pass

_maybe_extend_aiter_jit_path()
PY
    
    # vLLM/ROCm flags
    export VLLM_TARGET_DEVICE=\${VLLM_TARGET_DEVICE:-rocm}
    export VLLM_WORKER_MULTIPROC_METHOD=\${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
    export HIP_ARCHITECTURES=\${HIP_ARCHITECTURES:-gfx942}
    # Disable AIter Triton fused shared-experts path (avoids missing MI300X fused GEMM config JSONs).
    export VLLM_ROCM_USE_AITER_TRITON_FUSED_SHARED_EXPERTS=0

    # from https://rocm.docs.amd.com/en/docs-7.0-docker/benchmark-docker/inference-vllm-deepseek-r1-fp8.html
    # Note: this flag may not be compatible with MI325X GPUs
    # export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=\"\${VLLM_ROCM_QUICK_REDUCE_QUANTIZATION:-INT4}\"
    
    # Triton cache isolation (avoid multi-rank races on shared filesystems)
    # Use per-rank cache dirs on node-local /tmp.
    export TRITON_CACHE_DIR=\"/tmp/triton_cache/\${SLURM_JOB_ID:-nojob}/\${SLURM_PROCID:-\${SLURM_LOCALID:-0}}\"
    export XDG_CACHE_HOME=\"/tmp/xdg_cache/\${SLURM_JOB_ID:-nojob}/\${SLURM_PROCID:-\${SLURM_LOCALID:-0}}\"
    mkdir -p \"\$TRITON_CACHE_DIR\" \"\$XDG_CACHE_HOME\" 2>/dev/null || true

    # Create necessary directories
    export TORCH_EXTENSIONS_DIR=\"\${TORCH_EXTENSIONS_DIR:-/dev/shm/torch_ext}\"
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
