# CLAUDE.md - Repository-Wide Notes for AI Agents

## ROCm GPU Visibility in Singularity Containers (2026-04-21)

### Rule

When writing SLURM sbatch scripts that run GPU workloads inside Singularity/Apptainer containers with `--rocm`, **only set `HIP_VISIBLE_DEVICES`**. Never set `ROCR_VISIBLE_DEVICES`.

### Why

`ROCR_VISIBLE_DEVICES` filters at the HSA runtime agent level. HSA agent indices include CPU agents interspersed among GPU agents, so index N does not correspond to HIP device N. Setting `ROCR_VISIBLE_DEVICES=4,5,6,7` selects wrong/nonexistent HSA agents and produces `RuntimeError: No HIP GPUs are available`, even though the same GPUs work fine via `HIP_VISIBLE_DEVICES=4,5,6,7`.

`HIP_VISIBLE_DEVICES` filters at the HIP layer where indices match the user-facing GPU numbering (0-7 on an 8-GPU node). It works correctly for all GPU subsets.

### Evidence

Tested on TensorWave MI325X nodes (8 GPUs, 256 GiB each) with a 2-task sbatch job (`--gpus-per-node=8 --ntasks-per-node=2 --exclusive=user`) running `singularity exec --rocm --cleanenv`:

| Environment Variables (set inside container) | Task GPUs | Result                        |
|----------------------------------------------|-----------|-------------------------------|
| `HIP_VISIBLE_DEVICES=0,1,2,3`               | 0-3       | 4 GPUs working, 256 GiB each |
| `HIP_VISIBLE_DEVICES=4,5,6,7`               | 4-7       | 4 GPUs working, 256 GiB each |
| `ROCR_VISIBLE_DEVICES=0,1,2,3` + `HIP_VISIBLE_DEVICES=0,1,2,3` | 0-3 | Works (lucky — HSA agents 0-3 happen to be GPUs) |
| `ROCR_VISIBLE_DEVICES=4,5,6,7` + `HIP_VISIBLE_DEVICES=4,5,6,7` | 4-7 | **BROKEN** — "No HIP GPUs are available" |

### Correct Pattern for Multi-Task GPU Isolation

For `--ntasks-per-node=2` with `--gpus-per-node=8` (4 GPUs per task):

```bash
LOCALID=${SLURM_LOCALID:-0}
GPUS_PER_TASK=4
start_gpu=$(( LOCALID * GPUS_PER_TASK ))
GPU_IDS=""
for (( i=0; i<GPUS_PER_TASK; i++ )); do
    if [ -z "$GPU_IDS" ]; then
        GPU_IDS="$(( start_gpu + i ))"
    else
        GPU_IDS="${GPU_IDS},$(( start_gpu + i ))"
    fi
done
export HIP_VISIBLE_DEVICES="$GPU_IDS"
# Do NOT set ROCR_VISIBLE_DEVICES
```

### Affected Files

- `examples/autoif/execution/job_templates/dispatcher_slurm.sh.j2` — Jinja2 template for generated SLURM scripts
- `examples/lumi_task_launch.sh` — Reference launch script (already correct, uses HIP only)
- `examples/autoif/execution/environments/singularity_launcher.sh` — Container launcher library
- All generated scripts under `examples/autoif/data/*/generated_jobs/`
