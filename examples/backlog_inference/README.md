# Preemptible (Backfill) Inference

A shared example for running dispatcher-backed generation tasks over JSONL
input. It covers plain OpenAI-messages inference, two flavors of reasoning-trace
translation, and prompt/ground-truth translation. Workers are submitted as
SLURM jobs and can run as regular or preemptible backlog jobs — they pull work
from a central dispatcher server, so workers can come and go flexibly without
losing progress.

## TL;DR

The shortest path from "I have a JSONL file" to "results are being generated".
The five sections under [Structure](#structure) below cover anything not
spelled out here. The existing `configs/*.conf` files are tested, working
baselines — copy one and adapt.

### 1. Write a server config

Create `configs/dispatcher-server-<name>.conf` with at minimum:

```
INPUT_FILE=/abs/path/to/input.jsonl
OUTPUT_FILE=/abs/path/to/output.jsonl
SESSION_NAME=dispatcher-server-<name>   # defaults to dispatcher-server
```

- **`DISPATCHER_PKG`** defaults to two levels above the launch dir (i.e. the
  repo root, when the script is run from `examples/backlog_inference/`), so
  no config line is needed when running from the in-tree example. Override
  only if you want to point at a checkout outside this tree. The server
  installs the package on startup (`SKIP_DISPATCHER_INSTALL=0` default);
  workers default to `SKIP_DISPATCHER_INSTALL=1` because the server has
  already installed it — unset it on the worker side only if you want each
  worker to reinstall.
- **`WORK_TIMEOUT`** (default in the launcher is fine for most cases) is the
  per-item timeout and should be **calibrated to the task's generation
  length**: 32k+ token generations want **5000+ seconds**, short generations
  are fine **under 2000**. Too low and items get retried needlessly; too high
  and dead workers take forever to be re-scheduled.

Start the server one of two ways:

- **CPU login node** (fast, easy): `./start_dispatcher_server.sh configs/dispatcher-server-<name>.conf`.
  Risk: the server can OOM if pending writes back up past ~200k items — this
  happens on long-generation profiles where the writer cannot drain as fast
  as workers produce results.
- **GPU compute node via SLURM** (robust): `sbatch jobs/launch_dispatcher_server.sh configs/dispatcher-server-<name>.conf`.

### 2. Write a worker config

Create `configs/dispatcher-worker-<name>.conf` with at minimum:

```
SERVER_ADDRESS_FILE=.dispatcher-server-<name>    # defaults to .dispatcher-server; must match server's SESSION_NAME (prefixed with a dot)
TASK=tasks.inference_task.InferenceTask
MODEL=/abs/path/to/model
MAX_MODEL_LEN=8192
```

- **`SERVER_ADDRESS_FILE`** is the `.<SESSION_NAME>` file the server writes —
  workers read the server address from it.
- **`LAUNCHER`** defaults to `$WORK_DIR/tw_singularity_launcher.sh` (the
  TensorWave launcher shipped alongside this README), so no config line is
  needed when running from `examples/backlog_inference/`. Override only when
  pointing at a different launcher script (e.g. a LUMI-specific one).
- **`REQUEST_TIMEOUT`** and **`WORKERS`** (defaults in the launcher are fine
  for most cases) — `REQUEST_TIMEOUT` is the per-vLLM-request timeout and,
  like `WORK_TIMEOUT` on the server side, should be calibrated to generation
  length. The existing configs are good references for both.
- **`VLLM_EXTRA_ARGS`** lets you forward arbitrary vLLM CLI flags to the
  in-worker vLLM server. **The value must be quoted** because it contains
  spaces:

  ```
  VLLM_EXTRA_ARGS="--swap-space=0 --no-enable-prefix-caching"
  ```

  It is parsed with `shlex.split` on the Python side, so quoting works the
  shell way (`--some-flag="value with spaces"` is fine).

Generation parameters (`temperature`, `max_tokens`, `extra_body`, etc.) do
**not** go in the worker config — they live inside the task's `GEN_PARAMS`
dict and travel with each request. Edit the task to change them.

### 3. Reuse an existing task or write your own

The five tasks under `tasks/` cover plain inference, two reasoning-trace
translation variants, and prompt/ground-truth translation. To add a new task,
subclass `GeneratorTask` and use `self.build_result(...)` for the return
value — see [Tasks](#tasks) below.

### 4. Submit workers

Mix one non-preemptible batch with as many backlog jobs as you want:

```
# anchor batch (will not be preempted)
sbatch jobs/launch_dispatcher_worker.sh configs/dispatcher-worker-<name>.conf

# preemptible backlog scale-out
sbatch --qos=backlog --requeue --array=0-15 \
       jobs/launch_dispatcher_worker.sh configs/dispatcher-worker-<name>.conf
```

Done — the server writes results to `OUTPUT_FILE` as workers complete items.

## Structure

```
examples/backlog_inference/
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
| Annotations | `dispatcher-server-annotations.conf` | `dispatcher-worker-annotations.conf` | `InferenceTask` |
| Prompt translation | `dispatcher-server-prompt-translation.conf` | `dispatcher-worker-prompt-translation.conf` | `PromptTranslationTask` |
| SFT pipeline (stage 5) | `dispatcher-server-sft-pipeline-stage5.conf` | `dispatcher-worker-sft-pipeline-stage5.conf` | `InferenceTask` |
| Translation | `dispatcher-server-translation.conf` | `dispatcher-worker-translation.conf` | `ReasoningTranslationTask` |
| Translation (split traces) | `dispatcher-server-translation-split-traces.conf` | `dispatcher-worker-translation-split-traces.conf` | `ReasoningTranslationSplitTracesTask` |

Server configs set `INPUT_FILE`, `OUTPUT_FILE`, `SESSION_NAME`,
`DISPATCHER_PORT`, `WORK_TIMEOUT`, and `MAX_RETRIES`. Worker configs set
`SERVER_ADDRESS_FILE` (matching the server's `.<SESSION_NAME>` file), `TASK`,
`MODEL`, and any vLLM or launcher overrides. `DISPATCHER_PKG` and `LAUNCHER`
default to the in-tree repo root and `tw_singularity_launcher.sh`
respectively, so neither needs to be set when running from this directory.

The TensorWave launcher (`tw_singularity_launcher.sh`) accepts overrides such
as `LAUNCHER_IMG`, `LAUNCHER_HF_HOME`, `LAUNCHER_TORCHINDUCTOR_CACHE`,
`LAUNCHER_HF_HUB_OFFLINE`, `LAUNCHER_TRANSFORMERS_OFFLINE`, and
`LAUNCHER_MODELS_BIND_MODE`, so task-specific configs can change container
behavior without copying the worker script.

This example is configured for the **TensorWave cluster**. To run on **LUMI**,
point the `LAUNCHER` variable in your worker config to a LUMI-specific
launcher script instead of `tw_singularity_launcher.sh`.

## Usage

All commands are run from `examples/backlog_inference/`.

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
sbatch jobs/launch_dispatcher_worker.sh configs/dispatcher-worker-annotations.conf
```

Preemptible backlog workers as a job array:

```
sbatch --qos=backlog --requeue --array=0-15 \
       jobs/launch_dispatcher_worker.sh configs/dispatcher-worker-annotations.conf
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
