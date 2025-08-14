### Example scripts

This directory contains example code for working with dispatcher.
- `inference.py` simple batch inference script, usable as is for a variety of workflows
- `example_task.py` An example implementation of the more advanced multi-step inference interface.

## inference.py

Example usage of dispatcher server.

This is an implementation of a simple command line inference utility.  Simply
prepare your json, edit the sbatch config if necessary, and launch the script.

```bash
# download some data.
# we'll put it in the .messages[0].content format like many chat prompts are,
# though this is not at all necessary, you could just update the PROMPT_PATH in
# the sbatch config to be ".prompt", but I wanted to use a more complicate
# PROMPT_PATH for the demo.
python -c "import json, datasets; [(json.dump({'messages': [{'role': 'user', 'content': item['prompt']}]}, f), f.write('\n')) for item in datasets.load_dataset('argilla/prompt-collective')['train'] for f in [open('input.jsonl', 'a')]]"

# review settings in lumi_launch.sh and launch
sbatch lumi_launch.sh
```

The script will inference all the prompts.  If the job exits before the work is
done, requeuing the job will cause it to continue from where it left off.

The job will exit once all work is complete.


## example_task.py

This is an implemenetation of the `GeneratorTask` interface, which allows for
simple declarative definition of a multi-step inference workflow. The dispatcher
taskmanager executes many tasks simultaneously to keep the backend inference
servers busy.  A lower level `Task` class is also available, but for most use
cases you should use the `GeneratorTask` class instead.

Review the `example_task.py` code to see how a `GeneratorTask` should be
implemented.  All that's required is to subclass the `GeneratorTask` class, and
implement `task_generator` method.  The method should yield the requests it
needs to make until it is complete, then return the final results to complete
task execution.

The taskmanager framework handles fetching the work from the dispatcher
server, executing the queries, and reporting results back to the dispatcher
server.  The task code need only concern itself with the inference workflow.

To make working with tasks simple, we provide a simple cli class that handles
launching the vllm backend, and initializing the taskmanager configured to
retrieve tasks from a dispatcher backend.  For testing purposes, it also
supports a non-dispatcher workflow where only an input and output file are
specified, which is convenient for testing task implementations in interactive
sessions.

A few caveats for task design:
- Tasks are required to have work available (yield requests) immediately after creation.  This simplifies TaskManager's logic substantially, but means that if a task is created and does not yield requests to process, TaskManager will continue to request work and creating new tasks from the dispatcher server, in an attempt to keep the backend vllm server busy.
- Tasks are only intended to do lightweight processing between requests; if they run slowly, this will prevent TaskManager from keeping the backend vllm server busy.  This requirement could be removed with some modifications to the taskmanager architecture, but there has not yet been a need to do this work.


### Running the example task

As in the `inference.py` example above, we'll prepare the prompt dataset in a
standard multi-turn chat format, but for tasks, the input jsonl format is unconstrained,
and can contain whatever is useful or convenient.

```bash
python -c "import json, datasets; [(json.dump({'messages': [{'role': 'user', 'content': item['prompt']}]}, f), f.write('\n')) for item in datasets.load_dataset('argilla/prompt-collective')['train'] for f in [open('input.jsonl', 'a')]]"
```

Review the `lumi_task_launch.sh` to see how to launch task manager to execute
your task, configure it for your environment, and launch it.


# Development

Here is some information on developing new tasks.

## Development workflow

When developing a new `GeneratorTask`, the most efficient feedback loop involves running your task logic against a persistent, already-running model server. This avoids the long model-loading delay on every code change.

This recommended workflow uses two components:
1.  A standalone **vLLM server** running in one terminal.
2.  Your **task runner** process in a second terminal, operating in **local file mode**.

Notably, this setup **does not require** a `dispatcher-server`. The task runner reads directly from an input file, making iteration extremely fast.

### Step 0: Start an interactive session on a GPU node.

On LUMI, you might run something like this:

```bash
srun \
    --account=project_462000353 \
    --partition=dev-g \
    --ntasks=1 \
    --gres=gpu:mi250:8 \
    --time=4:00:00 \
    --mem=0 \
    --pty \
    bash
```


### Step 1: Start the vLLM Server

Launch the vLLM OpenAI-compatible server directly. You will only do this once per development session. This command tells `vllm` which model to load and where to listen for API requests.

You probably want to launch it in the background and potentially redirect its voluminous output to a log file.

You'll need to set options like `--tensor-parallel-size` correctly for your model.

```bash
# Activate your Python environment if needed, e.g.
module use /appl/local/csc/modulefiles; module load pytorch/2.7


# Launch the vLLM server and leave it running
MODEL=meta-llama/Llama-3.1-8B-Instruct
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1
```

*   `--model`: The Hugging Face model ID or path.
*   `--host 0.0.0.0`: Makes the server accessible from outside its container if needed. Use `127.0.0.1` to restrict it to the local machine.
*   `--port`: The port to listen on.
*   `--tensor-parallel-size`: The number of GPUs to use for the model.

The server will start and wait for requests.

### Step 2: Run Your Task Manager in Local File Mode (Terminal 2)

Run your task via the taskmanager CLI.  The key is to use the `--input` and `--output` flags to specify a local data source, and the `--no-launch` flag to connect to your existing vLLM server.

```bash
# Run your task, reading from a local file and connecting to the running server.
python -m dispatcher.taskmanager.cli \
    --task your_task_module.YourTaskClass \
    --input path/to/input.jsonl \
    --output path/to/output.jsonl \
    --model $MODEL \
    --host 127.0.0.1 \
    --port 8000 \
    --workers 1 \
    --no-launch
```

**Key Parameters:**
*   `--task`, `--input`, `--output`: Define your task and local data files. This enables the local file mode.
*   `--no-launch`: This prevents the script from trying to start a new vLLM process.
*   `--host` and `--port`: Must match the host and port of your running vLLM server from Terminal 1.
*   `--model`: This is still required. The task framework needs it to correctly format the API requests to the OpenAI API.
*   `--workers`: Limiting to a single worker can make reading log output more tractable while debugging.

Now, you can edit your task logic in `your_task_module.py` and re-run the `python -m dispatcher.taskmanager.cli` command instantly. You will get immediate feedback on your changes without waiting for the model to reload.

## Graceful Failure Reporting in Tasks

Sometimes, a task cannot be completed due to invalid input data or an unexpected result from an intermediate step. Instead of returning a custom error dictionary, the preferred way to handle this is to raise the `TaskFailed` exception.

This halts the task's execution immediately and ensures a standardized `__ERROR__` payload is written as the task's final result, which is invaluable for downstream monitoring and debugging.

**Example:**

Suppose a model is expected to return "A" or "B", but returns something else.

```python
# In your task's task_generator method:
from dispatcher.taskmanager.task import TaskFailed

# ... get a response from a model ...
judge_text = judge_resp.get_text()

if judge_text not in ["A", "B"]:
    # This is a controlled failure. Raise the exception.
    raise TaskFailed(
        message=f"Judge model returned an invalid response: '{judge_text}'",
        error_type="judge_response_invalid"
    )
```

This will produce the following clean, standardized error in your `output.jsonl` file, allowing you to easily find and analyze all failed tasks:

```json
{
  "__ERROR__": {
    "error": "judge_response_invalid",
    "message": "Judge model returned an invalid response: 'C'",
    "task_data": {
      "messages": [{"role": "user", "content": "..."}]
    }
  }
}
```


# Troubleshooting

Some common failure modes and how to resolve them.

## Understanding Tombstone Entries in Output

To prevent a single problematic line in your input file from stalling an entire
multi-node job, the Dispatcher server has a self-healing mechanism. If a work
item fails to complete and is reissued multiple times (by default, 3 times),
the server will stop trying and write a "tombstone" error to the corresponding
line in the output file.

This ensures the job can always make progress. If you see an entry like this in
your `output.jsonl`, it means the input line at the same position could not be
processed:

```json
{"__ERROR__": {"error": "max_retries_exceeded", "work_id": 1234,
"original_content": "{\"prompt\": \"This was the problematic input...\"}"}}
```

The original_content key contains the exact input that failed, which is useful
for debugging. You can control this behavior with the --max-retries flag when
starting the dispatcher-server.


## OOM errors

If you get OOM killed while VLLM is loading and you are using a converted
checkpoint from megatron, check that the checkpoint is sharded into multiple
pytorch files the way huggingface prefers it, rather than a single pytorch.bin.

LLM will use a lot of memory if there is only one large file, and if you are
loading multiple copies simlutaneously you may run into issues.

You can cause huggingface to reshard the model using a code snippet like this.

```python
from transformers import AutoModelForCausalLM

model_path="/scratch/project_xxx...."
save_path="./newdir"
model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16)
model.save_pretrained(save_path)
# you'll probably want to copy the tokenizer over too.
```


# Huggingface timeouts

If you are launching a large number of workers, you may be more likely to run
into timeouts checking huggingface for updated model versions.  If you have
already downlaoded the model, You can avoid this by directly referencing the
huggingface cached version fo the model, which you can find like this:

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(repo_id="meta-llama/Llama-2-7b-hf", filename="config.json")
print("Cache path:", path)
```
