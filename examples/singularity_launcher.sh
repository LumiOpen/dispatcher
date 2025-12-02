#!/bin/bash
# Launcher library for Dispatcher SLURM jobs with Singularity containers
# Source this file in your SLURM job scripts to avoid duplicating container setup code

###############################################################################
# Configuration Variables (can be overridden before sourcing this file)
###############################################################################

# Default container and cache paths
: "${LAUNCHER_IMG:=/scratch/project_462000353/containers/vllm_v10.1.1.sif}"
: "${LAUNCHER_HF_CACHE_PROJECT:=project_462000963}"
: "${LAUNCHER_PYEXEC_IN_IMG:=/opt/miniconda3/envs/pytorch/bin/python}"
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
  export HF_HOME="/scratch/${LAUNCHER_HF_CACHE_PROJECT}/hf_cache"
  export TRANSFORMERS_CACHE="$HF_HOME"
  export TORCHINDUCTOR_CACHE="/scratch/${LAUNCHER_HF_CACHE_PROJECT}/torch_inductor_cache"
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
  export AITER_INSTALL="/workspace/.aiter/jit/install"

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
  export SINGULARITYENV_PYTHONPATH="$PYUSERPKG:$AITER_INSTALL:\${PYTHONPATH-}"
  export SINGULARITYENV_DISPATCHER_SERVER="${DISPATCHER_SERVER}"
  export SINGULARITYENV_DISPATCHER_PORT="${DISPATCHER_PORT}"
  
  if [ -n "${HF_HUB_OFFLINE:-}" ]; then
    export SINGULARITYENV_HF_HUB_OFFLINE="$HF_HUB_OFFLINE"
  fi
  if [ -n "${TRANSFORMERS_OFFLINE:-}" ]; then
    export SINGULARITYENV_TRANSFORMERS_OFFLINE="$TRANSFORMERS_OFFLINE"
  fi

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
    -B "/scratch/${LAUNCHER_HF_CACHE_PROJECT}:/scratch/${LAUNCHER_HF_CACHE_PROJECT}:rw"
    -B "$PWD:/workspace"
  )
  if [ -f /usr/share/libdrm/amdgpu.ids ]; then
    BINDS+=(-B /usr/share/libdrm:/usr/share/libdrm:ro)
  fi

  echo "[singularity_launcher] Singularity environment setup complete."
}

###############################################################################
# SING_EXEC command [args...]
# Helper function to run commands inside the Singularity container
# Usage: SING_EXEC "command to run"
###############################################################################
SING_EXEC() {
  singularity exec --rocm --cleanenv "${BINDS[@]}" "$IMG" bash --noprofile --norc -c "$@"
}

###############################################################################
# install_dispatcher_packages
# Installs dispatcher and ninja in the container
###############################################################################
install_dispatcher_packages() {
  echo "[singularity_launcher] Installing dispatcher and ninja in container..."
  SING_EXEC "$PIP_IN_IMG install --user --upgrade 'git+https://github.com/LumiOpen/dispatcher.git' ninja"
  echo "[singularity_launcher] Package installation complete."
}

###############################################################################
# get_aiter_staging_script
# Returns the Python script for staging AIter module
# This script must be run inside the container on each worker node
###############################################################################
get_aiter_staging_script() {
  cat <<'AITER_SCRIPT'
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
AITER_SCRIPT
}


###############################################################################
# setup_cleanup_trap
# Sets up a cleanup trap to kill the server process on exit
###############################################################################
setup_cleanup_trap() {
  cleanup() {
    echo "[singularity_launcher] Cleaning up server PID ${srv_pid:-}"
    kill "${srv_pid:-0}" 2>/dev/null || true
  }
  trap cleanup EXIT
}

###############################################################################
# run_aiter_staging
# Runs the AIter staging script inside a worker container
# Must be called from within the worker environment (inside srun)
###############################################################################
run_aiter_staging() {
  echo "Staging AIter module..."
  # Use the centralized script definition to avoid duplication
  get_aiter_staging_script > /workspace/stage_aiter.py
  
  # Run the staging script with the Python from the container
  # Expects $PYEXEC_IN_IMG to be set in the worker environment
  "$PYEXEC_IN_IMG" /workspace/stage_aiter.py
}

###############################################################################
# setup_launcher_environment
# Complete environment setup (simplified helper)
# Combines: setup_singularity_environment + install + cleanup_trap
###############################################################################
setup_launcher_environment() {
  setup_singularity_environment
  install_dispatcher_packages
  setup_cleanup_trap
}

###############################################################################
# import_container_config
# Imports container configuration from parent environment into worker context
# Must be called inside srun workers to inherit parent container settings
###############################################################################
import_container_config() {
  # Import cache and compiler variables from parent environment
  export HF_HOME="${HF_HOME}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}"
  export TORCHINDUCTOR_CACHE="${TORCHINDUCTOR_CACHE}"
  export CC="${CC}"
  export CXX="${CXX}"
  export PYEXEC_IN_IMG="${PYEXEC_IN_IMG}"
}

###############################################################################
# setup_worker_environment
# Complete worker-side environment setup (for use inside srun workers)
# Sets up Python paths, cache variables, creates directories, and runs AIter staging
# This encapsulates all AITER-related setup so parent scripts don't need to know about it
###############################################################################
setup_worker_environment() {
  # Import container configuration from parent environment
  import_container_config
  
  # Setup Python environment (including AITER)
  export PYTHONUSERBASE="/workspace/pythonuserbase"
  export PATH="$PYTHONUSERBASE/bin:$PATH"
  export AITER_INSTALL="$HOME/.aiter/jit/install"
  export PYTHONPATH="$PYTHONUSERBASE/lib/python3.12/site-packages:$AITER_INSTALL:${PYTHONPATH-}"
  export PYTHONNOUSERSITE=
  
  # vLLM/ROCm flags
  export VLLM_USE_V1=1
  export VLLM_TARGET_DEVICE=rocm
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  export HIP_ARCHITECTURES=gfx90a
  
  # Create necessary directories
  export TORCH_EXTENSIONS_DIR=/dev/shm/torch_ext
  mkdir -p "$TORCH_EXTENSIONS_DIR" "$AITER_INSTALL/private_aiter/jit" 2>/dev/null || true
  
  # Run AIter staging automatically
  run_aiter_staging
}

echo "[singularity_launcher] Library loaded. Available functions:"
echo "  Host-side functions:"
echo "    - setup_singularity_environment"
echo "    - install_dispatcher_packages"
echo "    - setup_cleanup_trap"
echo "    - setup_launcher_environment (combines all 3 above)"
echo "    - SING_EXEC"
echo "  Worker-side functions (inside srun):"
echo "    - setup_worker_environment (complete worker setup including AITER)"
echo "  Low-level functions (used internally):"
echo "    - import_container_config"
echo "    - run_aiter_staging"
echo "    - get_aiter_staging_script"

