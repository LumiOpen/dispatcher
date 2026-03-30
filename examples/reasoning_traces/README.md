# Reasoning Traces

## Running translations

Translations use a dispatcher server/worker architecture. The server distributes
work items to SLURM worker nodes, each running a vLLM backend. Workers are
preemptable — if a node is preempted, in-flight items time out and are
automatically reissued.

### 1. Start the dispatcher server

Run on the login node (lightweight FastAPI process, managed in a tmux session):

```sh
./start_dispatcher_server.sh [config]
```

The server saves its address to a file so workers can discover it automatically.

Common operations:

```sh
./start_dispatcher_server.sh [config] --status   # check server
./start_dispatcher_server.sh [config] --stop     # stop server
```

### 2. Submit translation workers

Submit an array of SLURM worker jobs:

```sh
sbatch jobs/launch_translation_workers.sh [config]
```

By default this submits an array of workers (configured via `#SBATCH --array`
in the script). Override the array size on the command line:

```sh
sbatch --array=1-8%8 jobs/launch_translation_workers.sh [config]   # 8 workers
sbatch --array=1 jobs/launch_translation_workers.sh [config]        # single worker
```

### Configuration

Both scripts accept an optional config file (shell-sourceable `KEY=VALUE` pairs).
See `configs/` for examples:

- **Server config** — input/output files, port, timeouts, retry policy
- **Worker config** — server address file, model, batch size, vLLM parameters

Example with config files:

```sh
./start_dispatcher_server.sh configs/dispatcher-server-test-retry.conf
sbatch jobs/launch_translation_worker.sh configs/dispatcher-workers-test-retry.conf
```

### Monitoring

```sh
# Server status and progress
./start_dispatcher_server.sh [config] --status
curl http://<server-address>/status

# Server logs
tail -f logs/<session-name>.err

# Worker logs (SLURM array job)
tail -f logs/<job-id>_<array-id>.err
```

---

## Experiments: Finnish reasoning traces quality evaluation

1. Translate prompts (deepscaler) with one model - this should be fixed in one experiment (for now deepseekv3)
- DeepSeek-V3 translations at `data/default-train-sample-100_translations_DeepSeek-V3_fi.jsonl`

2. Use different candidate non-reasoning models to generate answers with reasoning traces (although answers are not interesting to us here - perhaps can be affected by the prompt but in this stage we let the model generate the answer as well). The input to the model is the prompt with the math question in finnish + guiding the model to generate reasoning traces

- Implemented in `tasks/traces_task.py`

**Generate traces with Qwen2.5-72B-Instruct on LUMI**
To run inference on LUMI with Qwen2.5-72B-Instruct model (works with vllm in LUMI module `pytorch/2.5`):

```sh
sbatch jobs/launch_traces_task.sh
```

**Generate traces with DeepSeek-V3 on VULTR**
```sh
sbatch jobs/launch_traces_task_deepseekv3.sh
```


3. Take a reasoning model (fixed) to get the answer given the translated prompt and generated reasoning trace. This can be one of deepseekv3, r1 or qwen3 (MoE) - for now we use Qwen3-30B-A3B-Thinking-2507 as the reasoning model.

- Implemented in `tasks/answering_given_traces_task.py`

```sh
sbatch jobs/launch_answer_given_traces_task_sing.sh
```

4. The accuracy of the problem solving across the models in (2) is a proxy for how well the models in (2) translated the traces.

run
```sh
python evaluate_prompts.py <path_to_generated_answers> /scratch/project_462000353/posttraining_data/DeepScaleR-Preview-Dataset/default-train-sample-100.jsonl
```

**Traces generated with Qwen2.5-72B-Instruct**

```sh
python evaluate_prompts.py data/default-train-sample-100_translations_DeepSeek-V3_fi_traces_Qwen2.5-72B-Instruct_fi_answers_Qwen3-30B-A3B-Thinking-2507_fi.jsonl /scratch/project_462000353/posttraining_data/DeepScaleR-Preview-Dataset/default-train-sample-100.jsonl
------
Accuracy: 63.25% (253/400)
Pass@4: 74.00% (74/100)
```

**Traces generated with DeepSeek-V3**

```sh
python evaluate_prompts.py data/default-train-sample-100_translations_DeepSeek-V3_fi_traces_DeepSeek-V3_fi_answers_Qwen3-30B-A3B-Thinking-2507_fi.jsonl /scratch/project_462000353/posttraining_data/DeepScaleR-Preview-Dataset/default-train-sample-100.jsonl
------
Accuracy: 69.50% (278/400)
Pass@4: 81.00% (81/100)
```

## Baseline 1: Generate solutions without traces

**Approach 1: prefill the input with empty <think></think> tags**

```sh
sbatch jobs/launch_answer_given_empty_traces_task_sing.sh
# runs tasks/answering_given_empty_traces_task.py
```

## Baseline generations

**Baseline 1 "Upper bound": English math question + English traces from R1**

1. Generate DeepSeek-R1 reasoning traces (Vultr)

```
sbatch launch_deepseekr1_docker_task.sh
```

2. Generate responses with Qwen3 (LUMI)

```
sbatch jobs/launch_answer_original_english_task_sing.sh
```

**Baseline 2 "Lower bound": Empty traces**

Generate responses with Qwen3 (LUMI)

```
sbatch jobs/launch_answer_given_empty_traces_task_sing.sh
```

**Baseline 3: Machine translated traces (generated by R1, translated by V3)**

1. Generate DeepSeek-R1 reasoning traces (Vultr) [same as B1]

```
sbatch launch_deepseekr1_docker_task.sh
```

2. Machine translate with Deepseek-V3:

```
sbatch launch_deepseekv3_docker_task.sh
```

3. Generate responses with Qwen3 (LUMI)

```
sbatch jobs/launch_answer_given_traces_task_sing.sh
```
