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
  export SINGULARITYENV_VLLM_USE_V1=1
  export SINGULARITYENV_VLLM_TARGET_DEVICE=rocm
  export SINGULARITYENV_VLLM_WORKER_MULTIPROC_METHOD=spawn
  export SINGULARITYENV_HIP_ARCHITECTURES=gfx90a
  
  # Worker environment variables (for use inside container)
  export SINGULARITYENV_TORCH_EXTENSIONS_DIR=/dev/shm/torch_ext
  export SINGULARITYENV_PYTHONNOUSERSITE=

  # Create stage_aiter.py script so it's available in the container
  get_aiter_staging_script > "${PWD:-$(pwd)}/stage_aiter.py"
}

###############################################################################
# get_binds
# Returns bind mount arguments as an array
# This function-based approach ensures BINDS are always available regardless of sourcing context
###############################################################################
get_binds() {
  local binds=(
    -B /scratch/project_462000353
    -B /flash/project_462000353
    -B /scratch/project_462000394/containers/for-turkunlp-team
    -B /pfs/lustrep3/scratch/project_462000394/containers/for-turkunlp-team
    -B /scratch/project_462000963:/scratch/project_462000963:rw
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
    "$PIP_IN_IMG install --user --upgrade 'git+https://github.com/LumiOpen/dispatcher.git' ninja"
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

# Check if destination already exists and points to the correct source
need_update = True
if os.path.lexists(dst):
    try:
        if os.path.islink(dst):
            # Check if symlink points to the correct source
            actual_target = os.readlink(dst)
            # Resolve relative symlinks
            if not os.path.isabs(actual_target):
                actual_target = os.path.normpath(os.path.join(os.path.dirname(dst), actual_target))
            # Check if the resolved target is the same as our source
            if os.path.exists(actual_target) and os.path.exists(so_src):
                try:
                    if os.path.samefile(actual_target, so_src):
                        print(f"[stage_aiter] Symlink already exists and points to correct source: {dst} -> {so_src}")
                        need_update = False
                    else:
                        print(f"[stage_aiter] Symlink exists but points to different source. Removing old symlink.")
                        os.remove(dst)
                except OSError:
                    # samefile can fail in some cases, just remove and recreate
                    print(f"[stage_aiter] Could not verify symlink target, removing and recreating.")
                    os.remove(dst)
            else:
                print(f"[stage_aiter] Symlink target or source missing, removing old symlink.")
                os.remove(dst)
        elif os.path.exists(dst):
            # It's a regular file, check if it's the same file or needs updating
            try:
                if os.path.samefile(dst, so_src):
                    print(f"[stage_aiter] Destination is already the same file as source: {dst}")
                    need_update = False
                else:
                    print(f"[stage_aiter] Destination exists but is different. Removing old file.")
                    os.remove(dst)
            except OSError:
                # samefile can fail, just remove and recreate
                print(f"[stage_aiter] Could not verify file, removing and recreating.")
                os.remove(dst)
    except Exception as e:
        # If anything goes wrong checking, just remove and recreate
        print(f"[stage_aiter] Error checking existing destination: {e}, removing and recreating.")
        try:
            os.remove(dst)
        except:
            pass

if need_update:
    try:
        os.symlink(so_src,dst)
        print(f"[stage_aiter] Symlinked: {dst} -> {so_src}")
    except OSError as e:
        # If symlink fails (e.g., cross-filesystem or file exists), try copy
        # But first check if they're the same file
        try:
            if os.path.exists(dst) and os.path.samefile(so_src, dst):
                print(f"[stage_aiter] Source and destination are the same file, skipping copy")
            else:
                shutil.copy2(so_src,dst)
                print(f"[stage_aiter] Copied: {so_src} -> {dst}")
        except (OSError, shutil.SameFileError) as copy_err:
            # If copy also fails because they're the same file, that's actually OK
            if "same file" in str(copy_err).lower() or isinstance(copy_err, shutil.SameFileError):
                print(f"[stage_aiter] Source and destination are the same file, no action needed")
            else:
                raise

# Ensure the file exists and is readable
if not os.path.exists(dst):
    raise SystemExit(f"[stage_aiter] FATAL: Destination file {dst} does not exist after symlink/copy")
if not os.access(dst, os.R_OK):
    raise SystemExit(f"[stage_aiter] FATAL: Destination file {dst} is not readable")

# Ensure inst_root is in sys.path (it should already be via PYTHONPATH, but be explicit)
if inst_root not in sys.path:
    sys.path.insert(0, inst_root)

# Clear any cached imports for this module to force a fresh import
module_name = "private_aiter.jit.module_aiter_enum"
if module_name in sys.modules:
    del sys.modules[module_name]
# Also clear parent modules if they exist
for mod in list(sys.modules.keys()):
    if mod.startswith("private_aiter"):
        del sys.modules[mod]

# Now import the module
m=importlib.import_module(module_name)
print(f"[stage_aiter] Successfully imported {module_name} from {m.__file__}")
print("[stage_aiter] Staging complete.")
AITER_SCRIPT
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
    
    # Setup Python environment
    export PYTHONUSERBASE=\"/workspace/pythonuserbase\"
    export PATH=\"\$PYTHONUSERBASE/bin:\$PATH\"
    export AITER_INSTALL=\"\$HOME/.aiter/jit/install\"
    export PYTHONPATH=\"\$PYTHONUSERBASE/lib/python${LAUNCHER_PYTHON_VERSION}/site-packages:\$AITER_INSTALL:\${PYTHONPATH-}\"
    export PYTHONNOUSERSITE=
    
    # vLLM/ROCm flags
    export VLLM_USE_V1=\${VLLM_USE_V1:-1}
    export VLLM_TARGET_DEVICE=\${VLLM_TARGET_DEVICE:-rocm}
    export VLLM_WORKER_MULTIPROC_METHOD=\${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
    export HIP_ARCHITECTURES=\${HIP_ARCHITECTURES:-gfx90a}
    
    # Create necessary directories
    export TORCH_EXTENSIONS_DIR=/dev/shm/torch_ext
    mkdir -p \"\$TORCH_EXTENSIONS_DIR\" \"\$AITER_INSTALL/private_aiter/jit\" 2>/dev/null || true
    
    # Define run_python helper function
    run_python() {
      \"\${PYEXEC_IN_IMG:-python3}\" \"\$@\"
    }
    
    # Run AIter staging automatically if script exists
    if [ -f /workspace/stage_aiter.py ]; then
      run_python /workspace/stage_aiter.py >&2 || true
    fi
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
