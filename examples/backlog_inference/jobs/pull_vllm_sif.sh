#!/bin/bash
#SBATCH --job-name=pull_vllm_sif
#SBATCH --partition=amd-tw-verification
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/pull_%j.out
#SBATCH --error=logs/pull_%j.err

set -euo pipefail

TAG="nightly_main_20260514"
DATESTAMP="20260515"
OUT_NAME="vllm-dev_${TAG}_${DATESTAMP}.sif"
DEST_DIR="/shared_silo/scratch/containers"
FALLBACK_DIR="/shared_silo/scratch/adamhrin@amd.com/containers"

# Pick singularity or apptainer
if command -v singularity >/dev/null 2>&1; then
    SING=singularity
elif command -v apptainer >/dev/null 2>&1; then
    SING=apptainer
else
    echo "ERROR: neither singularity nor apptainer on PATH" >&2
    module avail 2>&1 | head -50 >&2 || true
    exit 1
fi
echo "Using container runtime: $SING ($($SING --version))"

# Choose output directory
if touch "${DEST_DIR}/.write_test_${SLURM_JOB_ID}" 2>/dev/null; then
    rm -f "${DEST_DIR}/.write_test_${SLURM_JOB_ID}"
    OUT_DIR="${DEST_DIR}"
else
    echo "WARN: ${DEST_DIR} not writable; falling back to ${FALLBACK_DIR}"
    mkdir -p "${FALLBACK_DIR}"
    OUT_DIR="${FALLBACK_DIR}"
fi

OUT_PATH="${OUT_DIR}/${OUT_NAME}"
echo "Output will be: ${OUT_PATH}"

# tmp dirs (image layers can be huge — use node-local /tmp)
export SINGULARITY_TMPDIR="/tmp/${USER}/${SLURM_JOB_ID}"
export APPTAINER_TMPDIR="${SINGULARITY_TMPDIR}"
export SINGULARITY_CACHEDIR="${SINGULARITY_TMPDIR}/cache"
export APPTAINER_CACHEDIR="${SINGULARITY_CACHEDIR}"
mkdir -p "${SINGULARITY_TMPDIR}" "${SINGULARITY_CACHEDIR}"

cd "${OUT_DIR}"
echo "Pulling docker://rocm/vllm-dev:${TAG} -> ${OUT_PATH}"
time "${SING}" build "${OUT_NAME}" "docker://rocm/vllm-dev:${TAG}"

echo "Done. Final size:"
ls -lh "${OUT_PATH}"

# cleanup tmp
rm -rf "${SINGULARITY_TMPDIR}" || true
