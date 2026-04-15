# Dispatcher

A work-queue system for large-scale distributed LLM inference on HPC clusters.

Dispatcher solves the core problem of batch inference at scale: keeping hundreds of GPU workers busy with work, recovering from preemption and crashes without losing progress, and writing correct, resumable output without pre-partitioning your data.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [1. Simple batch inference](#1-simple-batch-inference)
  - [2. Multi-step workflows with TaskManager](#2-multi-step-workflows-with-taskmanager)
- [Writing Tasks](#writing-tasks)
  - [The GeneratorTask interface](#the-generatortask-interface)
  - [Parallel requests within a task](#parallel-requests-within-a-task)
  - [Controlled failure: TaskFailed](#controlled-failure-taskfailed)
  - [Retry on bad output: TaskRetry](#retry-on-bad-output-taskretry)
- [Running on a Cluster (LUMI / Slurm)](#running-on-a-cluster-lumi--slurm)
  - [Simple inference: lumi_launch.sh](#simple-inference-lumi_launchsh)
  - [Task-based inference: lumi_task_launch.sh](#task-based-inference-lumi_task_launchsh)
  - [GPU assignment on multi-task nodes](#gpu-assignment-on-multi-task-nodes)
  - [Preemption and signal handling](#preemption-and-signal-handling)
- [Developing and Testing Tasks Locally](#developing-and-testing-tasks-locally)
- [Dispatcher Server Reference](#dispatcher-server-reference)
  - [Server flags](#server-flags)
  - [API endpoints](#api-endpoints)
  - [Checkpoint and resume](#checkpoint-and-resume)
  - [Tombstone entries](#tombstone-entries)
  - [Updating work timeout live](#updating-work-timeout-live)
- [Dispatcher Client Reference](#dispatcher-client-reference)
- [TaskManager CLI Reference](#taskmanager-cli-reference)
- [Troubleshooting](#troubleshooting)

---

## How It Works

The dispatcher server reads a JSONL input file line by line and maintains a work queue. Workers poll the server for batches of items, process them, and submit results back. The server writes results to an output JSONL file, preserving 1-to-1 line correspondence with the input.

Key properties:

- **No pre-partitioning.** Workers pull work on demand. Slow items don't hold up fast ones. Heterogeneous processing times are handled naturally.
- **Crash recovery.** The server checkpoints progress. On restart, it picks up from the last written output line — no re-processing.
- **Preemption recovery.** Workers release their in-flight items back to the queue on `SIGTERM`. Other workers pick them up immediately without waiting for a timeout.
- **Fault tolerance.** Items that fail repeatedly are tombstoned (an `__ERROR__` line is written) so one bad input can't stall the entire job.
- **Ordered output.** Output line N always corresponds to input line N, even when workers complete items out of order.

---

## Architecture Overview

```
┌─────────────────────────────────────────┐
│          Dispatcher Server              │
│  (FastAPI, runs on head node)           │
│                                         │
│  input.jsonl  ──►  work queue           │
│                        │                │
│               GET /work│                │
│                        ▼                │
│              issued items (in-flight)   │
│                        │                │
│              POST /results              │
│                        │                │
│  output.jsonl  ◄── pending_write buf    │
│                   (ordered flush)       │
│                                         │
│  POST /release  ◄─── SIGTERM handler    │
└─────────────────────────────────────────┘
          ▲ ▲ ▲  workers poll concurrently
          │ │ │
   ┌──────┘ │ └──────┐
   │        │        │
Worker 0  Worker 1  Worker N
(node 0)  (node 0)  (node K)
vLLM TP=4  vLLM TP=4  vLLM TP=4
```

Each worker runs its own local vLLM instance (loaded once, reused for all items it processes). The dispatcher server is stateless with respect to the workers — any worker can process any item, and new workers can join mid-run.

---

## Installation

```bash
pip install -e .[dev]
```

Or from GitHub (e.g., inside a container):

```bash
pip install --user 'git+https://github.com/LumiOpen/dispatcher.git'
```

---

## Quick Start

### 1. Simple batch inference

Use `examples/inference.py` for straightforward prompt-in, response-out workflows.

**Prepare input** (one JSON object per line):
```bash
echo '{"messages": [{"role": "user", "content": "What is 2+2?"}]}' > input.jsonl
```

**Start the dispatcher server:**
```bash
dispatcher-server \
  --infile input.jsonl \
  --outfile output.jsonl \
  --port 9999
```

**Start a worker** (in another terminal or via `srun`):
```bash
python examples/inference.py \
  --model_path meta-llama/Llama-3.1-8B-Instruct \
  --dispatcher_server localhost:9999 \
  --batch_size 8
```

The worker loads vLLM, polls for batches, and submits results. The server exits automatically once all lines are processed. `output.jsonl` has one result per input line.

---

### 2. Multi-step workflows with TaskManager

For workflows that require multiple LLM calls per input item (generate → judge, generate → validate → retry, etc.), use the TaskManager with a `GeneratorTask`.

**Start the dispatcher server** (same as above).

**Run the task manager:**
```bash
dispatcher-task-run \
  --task examples.example_task.CompareTwoResponsesTask \
  --dispatcher localhost:9999 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel 1 \
  --workers 8
```

The task manager launches a local vLLM server, pulls work from the dispatcher, and runs your task logic. Each task instance handles one input item through all its steps, issuing as many LLM calls as needed before returning a result.

---

## Writing Tasks

### The GeneratorTask interface

Subclass `GeneratorTask` and implement `task_generator`. This is a Python generator — yield a `Request` (or list of `Request`s) to make LLM calls, receive `Response` objects back, and `return` a dict when done.

```python
from dispatcher.taskmanager.task import GeneratorTask, TaskFailed, TaskRetry
from dispatcher.taskmanager.backend.request import Request, Response

class MyTask(GeneratorTask):
    def task_generator(self):
        # self.data contains the parsed JSON from the input JSONL line
        messages = self.data["messages"]

        # Yield a request; execution suspends here until the response arrives.
        response: Response = yield Request({
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2048,
        })

        text = response.get_text()

        # Return the result dict — this becomes the output JSONL line.
        return {"response": text, "model": response.model_name}
```

**Rules for `task_generator`:**
- Must yield at least one `Request` before returning. The TaskManager requires work to be immediately available when a task is created. A task that returns without yielding will cause the manager to spin endlessly creating new tasks trying to fill the backend.
- Should only do lightweight processing between yields. Heavy CPU work between LLM calls stalls the scheduler and starves other concurrent tasks.
- Access your input data via `self.data` (the parsed JSON dict from the input JSONL line).

### Parallel requests within a task

Yield a list of `Request` objects to issue multiple calls simultaneously. Execution resumes when all of them complete.

```python
# Issue two generation requests in parallel, then judge them.
responses: list[Response] = yield [
    Request({"messages": messages, "temperature": 0.7, "max_tokens": 2048}),
    Request({"messages": messages, "temperature": 0.7, "max_tokens": 2048}),
]
resp_a, resp_b = responses
```

See `examples/example_task.py` (`CompareTwoResponsesTask`) for a complete example of this pattern including a judge step.

### Controlled failure: TaskFailed

When the input is invalid or an intermediate result makes completion impossible, raise `TaskFailed`. This writes a structured `__ERROR__` object to the output line and stops the task cleanly — no retry.

```python
judge_text = judge_resp.get_text().strip().upper()

if judge_text not in ("A", "B"):
    raise TaskFailed(
        message=f"Judge returned unexpected output: '{judge_text}'",
        error_type="invalid_judge_response",
    )
```

Output written to `output.jsonl`:
```json
{"__ERROR__": {"error": "invalid_judge_response", "message": "Judge returned unexpected output: 'C'", "task_data": {...}}}
```

### Retry on bad output: TaskRetry

When a response is structurally wrong and you want the item re-attempted (possibly on a different worker), raise `TaskRetry`. This releases the item back to the dispatcher's queue immediately with its full timeout window reset.

```python
text = resp.get_text()
if "ANSWER:" not in text:
    raise TaskRetry(message="Response missing required ANSWER: marker")
```

`TaskRetry` shares the same retry budget as timeout-based reissues. Once an item exceeds `max_retries` (default: 3 across all causes), it is tombstoned automatically. There is no risk of infinite loops.

See `examples/example_task.py` (`ValidatedResponseTask`) for a complete example.

---

## Running on a Cluster (LUMI / Slurm)

### Simple inference: lumi_launch.sh

Edit the configuration block at the top of `examples/lumi_launch.sh`:

```bash
INPUT_FILE=input.jsonl
OUTPUT_FILE=output.jsonl
PROMPT_PATH='.messages[0].content'    # jq-style path into each JSON line
MODE=chat                              # "chat" or "completion"

MODEL=meta-llama/Llama-3.3-70B-Instruct
GPUS_PER_TASK=4                        # GPUs per worker; match to model size
# --ntasks-per-node should be int(8 / GPUS_PER_TASK)

BATCH_SIZE=64
NUM_GENERATIONS=1
```

Submit:
```bash
sbatch examples/lumi_launch.sh
```

What the script does:
1. Starts the dispatcher server on the head node in the background.
2. Waits for the server to accept connections (up to 120 s).
3. Runs `srun` to launch one worker per Slurm task across all allocated nodes.
4. Each worker computes its GPU slice from `SLURM_LOCALID`, sets `HIP_VISIBLE_DEVICES`, and starts `inference.py`.

### Task-based inference: lumi_task_launch.sh

Edit the configuration block at the top of `examples/lumi_task_launch.sh`:

```bash
INPUT_FILE=input.jsonl
OUTPUT_FILE=output.jsonl
TASK=example_task.CompareTwoResponsesTask   # dotted path to your GeneratorTask subclass

WORKERS=32        # concurrent tasks per worker process
BATCH_SIZE=1      # work items pulled per dispatcher request (1 is usually correct)

REQUEST_TIMEOUT=600    # seconds before a single LLM call times out
WORK_TIMEOUT=1800      # seconds before dispatcher reissues an in-flight item

MODEL=meta-llama/Llama-3.3-70B-Instruct
GPUS_PER_TASK=4
MAX_MODEL_LEN=16384
```

Submit:
```bash
sbatch examples/lumi_task_launch.sh
```

### GPU assignment on multi-task nodes

When multiple Slurm tasks share a node (e.g., `--ntasks-per-node=2` with `--gpus-per-node=8`), the launch scripts use `SLURM_LOCALID` to assign non-overlapping GPU slices:

```bash
start_gpu=$(( SLURM_LOCALID * GPUS_PER_TASK ))
# LOCALID=0, GPUS_PER_TASK=4 → HIP_VISIBLE_DEVICES=0,1,2,3
# LOCALID=1, GPUS_PER_TASK=4 → HIP_VISIBLE_DEVICES=4,5,6,7
```

Adjust `--ntasks-per-node` to `int(8 / GPUS_PER_TASK)` when changing model size:

| Model size | Typical TP | ntasks-per-node |
|---|---|---|
| 8B | 1 | 8 |
| 34B | 2 | 4 |
| 70B | 4 | 2 |
| 122B+ | 8 | 1 |

### Preemption and signal handling

On LUMI, Slurm sends `SIGTERM` to the job when it is preempted. The `lumi_task_launch.sh` script uses a careful two-layer signal strategy to ensure in-flight work is released before the container shuts down:

```bash
# Layer 1 — outer bash: forward SIGTERM to the srun process group
trap '[ -n "$_CHILD_PID" ] && kill -TERM -- -"$_CHILD_PID"' TERM INT
set -m  # job control: srun runs in its own process group

srun bash -c "
  trap : TERM HUP   # Layer 2 — inner bash: ignore SIGTERM, let Python handle it

  run_python -m dispatcher.taskmanager.cli ...
"
```

The Python signal handler in `cli.py`:
1. Collects `work_id` from every active task's context.
2. Calls `POST /release` — the dispatcher marks those items as immediately reissuable.
3. Closes the vLLM backend.
4. Exits cleanly.

Other workers pick up the released items within seconds, not after `work_timeout`.

> **Why ignore SIGTERM in the inner shell?** Singularity tears down the container's network stack when the shell process exits on SIGTERM, which would prevent the HTTP `POST /release` from completing. Ignoring the signal in bash lets the Python process handle it while the container network is still alive.

---

## Developing and Testing Tasks Locally

The fastest development loop uses a persistent vLLM server and the file-based task source — no dispatcher server needed.

**Terminal 1: start vLLM once and leave it running**

```bash
# On LUMI, get an interactive GPU node first:
srun --account=project_462000353 --partition=dev-g \
     --ntasks=1 --gres=gpu:mi250:1 --time=4:00:00 --mem=0 --pty bash

# Start vLLM:
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 1
```

**Terminal 2: iterate on your task**

```bash
dispatcher-task-run \
    --task mymodule.MyTask \
    --input small_sample.jsonl \
    --output output.jsonl \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --host 127.0.0.1 --port 8000 \
    --no-launch \
    --workers 1
```

Re-edit `mymodule.py` and re-run the second command. The model does not reload between runs. `--workers 1` keeps log output sequential and readable.

Once your task is working correctly, switch to dispatcher mode:
```bash
# Replace --input/--output/--no-launch with:
--dispatcher HEAD_NODE:9999
# And remove --host/--port (the task manager will launch its own vLLM)
```

---

## Dispatcher Server Reference

### Server flags

```
dispatcher-server --infile INPUT --outfile OUTPUT [options]
```

| Flag | Default | Description |
|---|---|---|
| `--infile` | required | Input JSONL file |
| `--outfile` | required | Output JSONL file |
| `--checkpoint` | `<outfile>.checkpoint` | Checkpoint file for resume |
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8000` | Bind port |
| `--work-timeout` | `1200` | Seconds before in-flight item is reissued to another worker |
| `--max-retries` | `3` | Reissues before tombstoning an item (`-1` = infinite) |
| `--retry` | `300` | Seconds workers should wait when queue is temporarily empty |

### API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/work?batch_size=N` | Pull up to N work items |
| `POST` | `/results` | Submit completed items |
| `POST` | `/release` | Return in-flight items to queue immediately |
| `GET` | `/status` | Queue state (items issued, pending, next id, reissue count) |
| `POST` | `/work_timeout` | Update work timeout without restarting |

`GET /work` returns one of three statuses:
- `OK` — items included in response; process them and call `POST /results`
- `RETRY` — all input consumed but some items still in-flight; poll again in `retry_in` seconds
- `ALL_WORK_COMPLETE` — everything done; workers should exit

### Checkpoint and resume

The server writes a checkpoint file (default: `output.jsonl.checkpoint`) tracking how many lines have been successfully written. On restart with the same `--infile` and `--outfile`, it seeks to that position automatically and resumes. In-flight items that were never submitted are reissued.

Nothing special is needed to resume — just re-run the same `dispatcher-server` command.

### Tombstone entries

If an item is reissued `max_retries` times (from timeouts, `TaskRetry`, or worker crashes) without ever completing, the server writes a tombstone to preserve output line alignment:

```json
{"__ERROR__": {"error": "max_retries_exceeded", "work_id": 42, "original_content": "{...original input...}"}}
```

The `original_content` field contains the exact input line, which is useful for debugging and reprocessing.

### Updating work timeout live

If inferences are taking longer than expected and items are being reissued prematurely:

```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"timeout": 3600}' \
     http://HEAD_NODE:PORT/work_timeout
```

This takes effect immediately without restarting the server or losing any in-flight state.

---

## Dispatcher Client Reference

Use `WorkClient` directly if you're writing a custom worker (e.g., using a non-vLLM inference backend or building your own batching logic):

```python
import time
import json
from dispatcher.client import WorkClient
from dispatcher.models import WorkStatus

client = WorkClient("http://head-node:9999")

while True:
    resp = client.get_work(batch_size=8)

    if resp.status == WorkStatus.ALL_WORK_COMPLETE:
        print("Done.")
        break

    elif resp.status == WorkStatus.RETRY:
        time.sleep(resp.retry_in)
        continue

    elif resp.status == WorkStatus.SERVER_UNAVAILABLE:
        # Server has exited (all work complete), or network issue.
        break

    elif resp.status == WorkStatus.OK:
        for item in resp.items:
            # item.content is always a raw string — parse if JSON.
            data = json.loads(item.content)
            # ... process data ...
            item.set_result(json.dumps({"output": "..."}))

        client.submit_results(resp.items)
```

To release items back to the queue (e.g., in a SIGTERM handler):

```python
work_ids = [item.work_id for item in in_flight_items]
client.release_work(work_ids)
```

---

## TaskManager CLI Reference

```
dispatcher-task-run --task MODULE.CLASS --model MODEL [source] [vllm] [manager]
```

**Source (mutually exclusive — pick one):**

| Flag | Description |
|---|---|
| `--dispatcher HOST:PORT` | Pull work from dispatcher server (distributed mode) |
| `--input PATH` | Read from local JSONL file (development mode) |

`--output PATH` is required with `--input`.

**vLLM:**

| Flag | Default | Description |
|---|---|---|
| `--model` | required | HF model ID or local path |
| `--host` | `127.0.0.1` | vLLM bind host |
| `--port` | `8000` | vLLM bind port |
| `--no-launch` | false | Connect to an already-running vLLM server |
| `--tensor-parallel` | `1` | Tensor parallel degree |
| `--max-model-len` | `16384` | Max context length override |
| `--startup-timeout` | `1500` | Seconds to wait for vLLM to start |
| `--request-timeout` | `600` | Seconds before a single LLM request times out |
| `--enforce-eager` | false | Disable CUDA/HIP graph capture |
| `--vllm-extra-args` | — | Extra args passed verbatim to vLLM (shell-quoted string) |
| `--silence-vllm-logs` | false | Suppress vLLM log output |

**TaskManager:**

| Flag | Default | Description |
|---|---|---|
| `--workers` | `16` | Max concurrent in-flight tasks |
| `--batch-size` | `4` | Items fetched per `GET /work` call |

**Example with extra vLLM args:**
```bash
dispatcher-task-run \
    --task mymodule.MyTask \
    --dispatcher head-node:9999 \
    --model Qwen/Qwen3-30B \
    --tensor-parallel 4 \
    --vllm-extra-args "--enable-thinking --trust-remote-code" \
    --workers 32
```

---

## Troubleshooting

### Workers exit immediately with "Server is unavailable"

The server exits automatically when all work is complete. If workers connect after the server has already exited (because a previous run finished all lines), they'll see `SERVER_UNAVAILABLE` and exit immediately. To reprocess, delete or rename the output file and checkpoint file, then restart the server.

### Items are being reissued constantly

Check `GET /status` — if `expired_reissues` is climbing, inferences are taking longer than `work_timeout`. Increase it live:

```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"timeout": 3600}' http://HEAD_NODE:PORT/work_timeout
```

Then increase `--work-timeout` in your Slurm script for future runs.

### Output has many tombstone `__ERROR__` lines

Tombstones appear when items are reissued `max_retries` times without completing. Common causes:
- Malformed input that always fails to parse.
- A required field is missing and the task always raises `TaskFailed`.
- `REQUEST_TIMEOUT` is shorter than actual inference time.

To reprocess tombstoned lines: extract them from the output (filter on `__ERROR__`), fix the inputs, and run a new dispatcher job on just those lines.

### OOM when loading the model

If vLLM OOMs during load and you are using a checkpoint converted from Megatron, check that weights are sharded into multiple `.safetensors` files rather than a single `pytorch_model.bin`. A single large file is loaded into CPU memory before being split across GPUs. Reshard with:

```python
from transformers import AutoModelForCausalLM
import torch
model = AutoModelForCausalLM.from_pretrained("/path/to/model", device_map="auto", torch_dtype=torch.bfloat16)
model.save_pretrained("./resharded")
# Copy the tokenizer too:
# from transformers import AutoTokenizer
# AutoTokenizer.from_pretrained("/path/to/model").save_pretrained("./resharded")
```

### HuggingFace timeouts on large jobs

When launching many workers simultaneously, all of them hit the HF Hub to check for model updates at the same time. If the model is already cached locally, set:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Or reference the local cache path directly as `--model`:

```python
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id="meta-llama/Llama-3.1-8B-Instruct", filename="config.json")
print(path)  # pass the parent directory as --model
```

### TaskManager spins without making progress

This happens when a `GeneratorTask` subclass doesn't yield any `Request` before returning. The TaskManager creates a task, sees no work queued, and immediately creates another. Ensure `task_generator` always yields at least one `Request` before any conditional return path.

### Output file stops growing despite workers running

The server buffers out-of-order completions and only flushes contiguous results. If one item is stuck (very slow inference, or waiting for timeout after a crash), all later results pile up in memory and the output file stops growing. Check `GET /status`:

- If `pending` is large but `issued` is small, one item is the bottleneck.
- Wait for it to timeout and be reissued, or increase `--work-timeout` if it just needs more time.
- If a worker has crashed holding that item, it will be reissued automatically after `work_timeout` seconds.
