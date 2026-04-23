# Inference Example

A simple example of running distributed inference over OpenAI-messages-formatted
JSONL input using the dispatcher framework. Workers are submitted as SLURM jobs
and can run as regular or preemptible backlog jobs — they pull work from a
central dispatcher server, so workers can come and go flexibly without losing
progress.

## Structure

- `configs/` — Configuration files for the server and workers.
- `tasks/inference_task.py` — The task definition. Each sample's `messages`
  array is sent to the model and the response is appended as an assistant turn.
- `jobs/launch_dispatcher_worker.sh` — SLURM job script that starts a vLLM
  backend and a dispatcher worker inside a Singularity container.
- `start_dispatcher_server.sh` — Starts the dispatcher server in a **tmux
  session on the login node**. The server is a lightweight FastAPI process that
  coordinates work distribution, tracks progress via a checkpoint file, and
  writes completed results to the output JSONL.
- `extract_errors.py` — Post-run utility for inspecting errored samples (see
  below).

## Configuration

Create your own config files based on the examples in `configs/`. The config
files are simple `KEY=VALUE` shell files — see the existing ones for the
available options. At minimum you'll want to set:

- **Server config** — `INPUT_FILE`, `OUTPUT_FILE`, and `DISPATCHER_PKG`
  pointing to your own directories.
- **Worker config** — `DISPATCHER_PKG` and `LAUNCHER` pointing to your own
  directories.

You'll also need to adjust the Singularity image path and bind-mount paths in
your launcher script (`tw_singularity_launcher.sh`) to match your environment.

Then pass your configs when starting the server and submitting workers (see
commands below).

This example is currently configured for the **TensorWave cluster**. To run on
**LUMI**, point the `LAUNCHER` variable in your worker config to a
LUMI-specific launcher script (e.g. `lumi_singularity_launcher.sh`) instead of
`tw_singularity_launcher.sh`.

## Usage

### 1. Start the dispatcher server

```
cd examples/inference
./start_dispatcher_server.sh configs/dispatcher-server-annotations.conf
```

This creates a background tmux session (`dispatcher-server-annotations`) on
the login node. Use `--status` / `--stop` to manage it.

### 2. Submit workers

Submit a single worker:

```
sbatch jobs/launch_dispatcher_worker.sh configs/dispatcher-workers-annotations.conf
```

Submit preemptible backlog workers as a job array:

```
sbatch --qos=backlog --requeue --array=0-15 jobs/launch_dispatcher_worker.sh configs/dispatcher-workers-annotations.conf
```

Workers are versatile — the same script works with both submission methods. Each
worker spins up a local vLLM instance, connects to the dispatcher server, and
pulls work items until none remain. Because all state lives on the server side,
workers can be preempted, requeued, or manually cancelled at any time without
data loss. As long as the server is running, you can freely add or remove
workers to scale throughput up or down.

### 3. Monitor

Check server status and progress:

```
./start_dispatcher_server.sh configs/dispatcher-server-annotations.conf --status
```

Server logs: `logs/dispatcher-server-annotations.*`

Worker/vLLM logs: `logs/{SLURM_JOBID}_{SLURM_ARRAY_TASK_ID}.*`

## Debugging errors

Use `extract_errors.py` to pull out failed samples from the output JSONL:

```
python extract_errors.py data/sft-pipeline-stage3-train.jsonl
```

This produces a `*.errors.json` file with the `prompt_id`, error reason, and
the original messages for each failed sample. In practice, nearly all errors
are `max_retries_exceeded` caused by samples with extremely long input content
(e.g. huge inline SVG blobs, long base64/hex dumps, or massive code pastes)
that exceed the model's context window or cause generation timeouts. These are
essentially data-quality issues in the input dataset rather than infrastructure
failures. 

### Solution

Increase the `MAX_MODEL_LEN` in `configs/dispatcher-workers-annotations.conf` to a higher number, e.g. 32768 and rerun for the errored samples

