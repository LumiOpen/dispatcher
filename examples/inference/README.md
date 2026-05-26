# Dispatcher Inference Example

A shared example for running dispatcher-backed generation tasks over JSONL
input. It covers plain OpenAI-messages inference, two flavors of reasoning-trace
translation, and prompt/ground-truth translation. Workers are submitted as
SLURM jobs and can run as regular or preemptible backlog jobs — they pull work
from a central dispatcher server, so workers can come and go flexibly without
losing progress.

## Structure

```
examples/inference/
├── README.md
├── start_dispatcher_server.sh      # login-node tmux launcher for the server
├── tw_singularity_launcher.sh      # vLLM + worker launcher (TensorWave)
├── configs/                        # KEY=VALUE shell configs per profile
├── jobs/
│   ├── launch_dispatcher_server.sh # sbatch wrapper to run the server on a compute node
│   ├── launch_dispatcher_worker.sh # sbatch worker job (vLLM + dispatcher worker)
│   └── pull_vllm_sif.sh            # one-time pull of the vLLM Singularity image
└── tasks/
    ├── inference_task.py
    ├── inference_task_think.py
    ├── prompt_translation_task.py
    ├── reasoning_translation_task.py
    └── reasoning_translation_split_traces_task.py
```

### Tasks

All five tasks subclass `dispatcher.taskmanager.task.base.GeneratorTask` and use
the standard result contract:

- `self.build_result(**payload)` to construct the return dict — spreads
  `self.data`, adds `success: True`, merges your payload, and (when retry
  metadata is available) includes `retry_count` and `max_retries`.
- `self.is_last_retry_attempt()` to detect the final retry attempt, so the
  task can write a partial / errored result instead of raising another
  `TaskRetry` that would become a dispatcher tombstone.

The translation tasks (`prompt_translation_task`, `reasoning_translation_task`,
`reasoning_translation_split_traces_task`) wrap `build_result` in a small local
`_failed_result` helper that adds per-task warning logs, then delegates to the
base helper for the schema.

| Task | Input shape | Output additions |
| --- | --- | --- |
| `InferenceTask` | `messages: [...]` | `response`, `output_messages` |
| `InferenceTask` (think variant) | `messages: [...]` | `response`, `output_messages` — uses larger context + Qwen-style thinking |
| `PromptTranslationTask` | `prompt`, `ground_truth` | `translated_prompt`, `translated_ground_truth` |
| `ReasoningTranslationTask` | `input` (messages), `output` (with `<think>` tags) | `translated_prompt`, `translated_traces` |
| `ReasoningTranslationSplitTracesTask` | same as above | same, but trace body and answer translated separately then recombined |

### Configs

Per profile there is one server config plus one worker config:

| Profile | Server config | Worker config | Task |
| --- | --- | --- | --- |
| Annotations | `dispatcher-server-annotations.conf` | `dispatcher-workers-annotations.conf` | `InferenceTask` |
| Prompt translation | `dispatcher-server-prompt-translation.conf` | `dispatcher-worker-prompt-translation.conf` | `PromptTranslationTask` |
| SFT pipeline (stage 5) | `dispatcher-server-sft-pipeline-stage5.conf` | `dispatcher-workers-sft-pipeline-stage5.conf` | `InferenceTask` |
| Translation | `dispatcher-server-translation.conf` | `dispatcher-workers-translation.conf` | `ReasoningTranslationTask` |
| Translation (split traces) | `dispatcher-server-translation-split-traces.conf` | `dispatcher-worker-translation-split-traces.conf` | `ReasoningTranslationSplitTracesTask` |

Server configs set `INPUT_FILE`, `OUTPUT_FILE`, `SESSION_NAME`,
`DISPATCHER_PORT`, `WORK_TIMEOUT`, `MAX_RETRIES`, and `DISPATCHER_PKG`. Worker
configs set `SERVER_ADDRESS_FILE` (matching the server's `.<SESSION_NAME>`
file), `TASK`, `MODEL`, `LAUNCHER`, and any vLLM or launcher overrides.

The TensorWave launcher (`tw_singularity_launcher.sh`) accepts overrides such
as `LAUNCHER_IMG`, `LAUNCHER_HF_HOME`, `LAUNCHER_TORCHINDUCTOR_CACHE`,
`LAUNCHER_HF_HUB_OFFLINE`, `LAUNCHER_TRANSFORMERS_OFFLINE`, and
`LAUNCHER_MODELS_BIND_MODE`, so task-specific configs can change container
behavior without copying the worker script.

This example is configured for the **TensorWave cluster**. To run on **LUMI**,
point the `LAUNCHER` variable in your worker config to a LUMI-specific
launcher script instead of `tw_singularity_launcher.sh`.

## Usage

All commands are run from `examples/inference/`.

### 1. Pull the vLLM image (once)

```
./jobs/pull_vllm_sif.sh
```

### 2. Start the dispatcher server

There are two ways to run the server. **Pick one** per profile.

#### Option A: login-node tmux session

Fast to start, fine for short-running jobs and ad-hoc work. The server lives
in a tmux session named after `SESSION_NAME` and can be inspected with
`--status` / stopped with `--stop`.

```
./start_dispatcher_server.sh configs/dispatcher-server-annotations.conf
./start_dispatcher_server.sh configs/dispatcher-server-annotations.conf --status
./start_dispatcher_server.sh configs/dispatcher-server-annotations.conf --stop
```

#### Option B: compute-node sbatch job

Preferred for long-running profiles where login-node uptime is unreliable, or
when the server should not share resources with interactive use. The job
script is a thin wrapper that calls `start_dispatcher_server.sh ... --run-server`
inside an allocation.

```
sbatch jobs/launch_dispatcher_server.sh configs/dispatcher-server-annotations.conf
```

The server process is identical between the two options — workers connect via
the `.<SESSION_NAME>` address file regardless of where the server runs.

### 3. Submit workers

A single worker:

```
sbatch jobs/launch_dispatcher_worker.sh configs/dispatcher-workers-annotations.conf
```

Preemptible backlog workers as a job array:

```
sbatch --qos=backlog --requeue --array=0-15 \
       jobs/launch_dispatcher_worker.sh configs/dispatcher-workers-annotations.conf
```

The same script works with both submission modes. Each worker spins up a local
vLLM instance, connects to the dispatcher server, and pulls work items until
none remain. Because all state lives on the server side, workers can be
preempted, requeued, or cancelled at any time without data loss. Add or remove
workers freely while the server is running.

Swap the config path to run any other profile, e.g.:

```
sbatch jobs/launch_dispatcher_worker.sh configs/dispatcher-worker-translation-split-traces.conf
```

### Monitor

```
./start_dispatcher_server.sh configs/dispatcher-server-annotations.conf --status
```

- Server logs (login-node mode): `logs/dispatcher-server-<session>.*`
- Server logs (compute-node mode): `logs/dispatcher-server-<jobid>.{out,err}`
- Worker / vLLM logs: `logs/<jobid>_<array_task_id>.*`

## Workaround: TensorWave epilog kills sibling jobs

> **Temporary — TensorWave only.** Remove this workflow once the cluster
> epilog at `/itshared/silo16/slurm_config/epilog.sh` is fixed.

When any SLURM job of yours ends on a TensorWave node, the cluster epilog
SIGKILLs every other process owned by your UID on that node — including your
dispatcher server and any other running jobs of yours. Root cause: the epilog
checks for sibling jobs at the cgroup v1 path
(`/sys/fs/cgroup/cpuset/slurm/uid_$UID/job_*`), but TensorWave is on cgroup
v2. The directory is missing, the sibling-detection loop sees an empty list,
and the script falls through to `kill -9` on every "leftover" process owned
by your UID. Those processes are not leftovers.

### Recommended workflow until the epilog is fixed

The idea is to fully occupy the server's node with your own non-preemptible
workers, so no shorter sibling job of yours can land there and trigger the
epilog kill.

1. Start the dispatcher server as a SLURM job (Option B above):
   ```
   sbatch jobs/launch_dispatcher_server.sh configs/dispatcher-server-annotations.conf
   ```

2. Note which node the server landed on:
   ```
   squeue --me -n dispatcher-server -o '%i %N %T'
   ```

3. Submit **8 non-preemptible workers pinned to that node**. With one GPU per
   worker this fills the node:
   ```
   sbatch --nodelist=<server-node> --array=0-7 \
          jobs/launch_dispatcher_worker.sh configs/dispatcher-workers-annotations.conf
   ```

4. Submit any additional workers as preemptible backlog jobs on other nodes —
   if one of these is preempted or finishes, the epilog only affects that
   other node, not the server's:
   ```
   sbatch --qos=backlog --requeue --array=0-15 \
          jobs/launch_dispatcher_worker.sh configs/dispatcher-workers-annotations.conf
   ```

**Trade-off**: one full GPU node is held by the inference workload for the
duration of the run. The benefit is that the server and its co-located
workers cannot be killed by another short job of yours ending on that node.
