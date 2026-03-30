# AutoIF pipeline

Generating synthetic instruction following data on SLURM environment, inspired by [AutoIF](https://github.com/QwenLM/AutoIF). Currently includes experiments for English and Finnish.

Pipeline overview:
1. Generate augmented constraints given seed constraints (latest seed data is IFEval + Tulu3-IFEval-OOD categorised into 4 categories `data/categorised_instructions.json`)
2. Generate python verifiers that verify whether a response adheres to the constraint (IFEval-like)
3. Construct prompts (can be multiturn) using the constraints and a user query (latest user queries taken from [lmsys-chat](https://huggingface.co/datasets/lmsys/lmsys-chat-1m))
4. Generate answers to the prompts, verify the constraints using the python verifiers and score the relevancy to the user's query with LLM-as-a-judge
5. Build final SFT dataset with the highest quality data

## Quick start on LUMI

1. Copy `config.lumi.yaml` and `slurm.lumi.yaml` into an experiment directory under `data/experiment_name` and edit it to set your parameters
2. Run the pipeline

```sh
python3 pipeline.py --config data/experiment_name/config.lumi.yaml --slurm-config data/experiment_name/slurm.lumi.yaml --out-dir data/experiment_name
```

## Quick start of VULTR - DeepSeek-V3

1. Copy `config.vultr.yaml` into an experiment directory under `data/experiment_name` and edit it to set your parameters
2. Submit a job that starts a vllm docker container and spawns a vllm backend with DeepSeek-V3. Once server is started the pipeline is run in an interactive mode

```sh
sbatch launch_deepseekv3_pipeline.sh
```

## How this pipeline works

The order of the jobs is configured in config.yaml under `pipeline`. Set true/false to enable/disable each job.:

```yaml
pipeline:
  - augmentation_preprocessing: true
  - augmentation_generation: true
  - augmentation_postprocessing: true
  - verifiers_preprocessing: true
  - verifiers_generation: true
  - cross_validation_job: true
  - concatenation_job: true
  - responses_generation: true
  - final_dataset_job: true
```

A CPU job specifies the script to run and its arguments

```yaml
augmentation_preprocessing:
  type: cpu_script
  script: src/create_instructions_input.py
  args:
    seed-file: "/scratch/project_462000353/adamhrin/dispatcher/examples/autoif/data/categorised_instructions_fi.json"
    output-file: "aug_input.jsonl"
    num-instructions-per-category: 50
    language: "${language}"
```

A GPU inference job in its simplest form specifies the dispatcher task implementation, input and output jsonl files

```yaml
augmentation_generation:
  type: dispatcher_task
  task: tasks.augmentation_task.AugmentInstructionsTask
  input_file: "aug_input.jsonl"
  output_file: "aug_output.jsonl"
```

**Tasks** in `tasks/` are dispatcher GeneratorTask implementations. This is the main logic of the LLM generations.

### Executing as sbatch queued jobs with dependencies (Large-scale generations)

Simply run

```sh
python3 pipeline.py --config data/experiment_name/config.lumi.yaml --slurm-config data/experiment_name/slurm.lumi.yaml --out-dir data/experiment_name
```

The pipeline generates job sbatch files using the templates at `execution/job_templates/` and environment setup at `execution/environments` into `data/experiment_name/generated_scripts` and submits them with dependencies (sequentially) so that input from one job is ready before starting the next job.

To configure slurm for each job modify `data/experiment_name/slurm.lumi.yaml`

### Executing on an interactive node (Experimentation, debugging)

1. Allocate a GPU node and run a vllm backend with your model of choice, for example with

```bash
# Activate your Python environment if needed, e.g.
module use /appl/local/csc/modulefiles; module load pytorch/2.5


# Launch the vLLM server and leave it running
MODEL=meta-llama/Llama-3.1-8B-Instruct
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1
```

2. Uncomment this section in your `config.yaml`

```yaml
vllm_server:
  host: "127.0.0.1"
  port: 8000
```

3. Run the pipeline

```sh
python3 pipeline.py --config data/experiment_name/config.lumi.yaml --slurm-config data/experiment_name/slurm.lumi.yaml --out-dir data/experiment_name
```


This tells the pipeline that you want to run all GPU jobs against this vllm backend and will run them with dispatcher in local file mode. The generated scripts are based on the template `execution/job_templates/dispatcher_local_job.sh.j2`


**Using pipeline flags**
```sh
python3 pipeline.py <args> [ --status ] # optional --status flag, check the status of the pipeline if not run for the first time
python3 pipeline.py <args> --force # rerun all steps from the beginning.
python3 pipeline.py <args> --rerun-failed # restart failed jobs and all subsequent jobs (cancel pending jobs if exists).
python3 pipeline.py <args> --continue # continue from the last failed job without rerunning it. Useful if partial result is enough
```

**Using custom template**
To add a custom job template, create a new file under `execution/job_templates/` and specify its path in the slurm config for the job, e.g.

```yaml
augmentation_generation:
  template: "execution/job_templates/my_custom_template.sh.j2"
```

**Using custom script**

If you don't want to use the template rendering system, you can provide a fully self-contained custom script:

```yaml
augmentation_generation:
  custom_script: "custom_jobs/my_custom_script.sh"
```

The custom script should be a complete, executable bash script ready to run. It will be executed directly without any parameter substitution or template rendering.

## Cross-validation job

The `cross_validation_job` validates LLM-generated Python verifier functions against
test cases on a CPU node. It runs as a standalone step after `verifiers_generation`.

### How to run

Cross-validation is part of the pipeline — enable it in your config:

```yaml
pipeline:
  - cross_validation_job: true
```

Or run standalone:

```bash
cd /scratch/project_462000963/users/adamhrin/dispatcher/examples/autoif
module load cray-python
pip install --user "transformers<4.48.0"
python3 src/verifiers_cross_validation.py \
  --input-file data/experiment/verifiers_output.jsonl \
  --output-file data/experiment/verifiers_validated.jsonl
```

### Pre-download requirements

<!-- TODO: Test downloading models directly on compute nodes instead of this
     two-step login-node pre-download. If reliable, refactor model_preloader
     to download on the compute node and remove this manual step. -->

Downloading models on compute nodes is slow and unreliable under load.
Pre-download all NLP models from the login node before submitting:

```bash
module load cray-python

# xlm-roberta-base (required by trankit)
HF_HOME=/scratch/project_462000963/cache python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('xlm-roberta-base', cache_dir='/scratch/project_462000963/cache/hub')
"

# spaCy Finnish model
PYTHONUSERBASE=$(pwd)/pythonuserbase python3 -m spacy download fi_core_news_sm
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `FUNCTION_TIMEOUT` | 10 | Timeout for function init (seconds) |
| `INIT_TIMEOUT` | FUNCTION_TIMEOUT | Timeout for exec/import phase |
| `CASE_TIMEOUT` | 30 | Per-test-case timeout |
| `MIN_FUNCTIONS` | 1 | Minimum passing functions required |
| `MIN_TEST_CASES` | 1 | Minimum passing test cases required |
| `FUNCTION_PASS_RATE` | 0.8 | Accuracy threshold for functions |
| `MIN_CASE_PASSES` | 1 | Minimum functions a case must pass |
| `CROSSVAL_WORKERS` | 32 | Number of worker processes |
| `THREADS_PER_WORKER` | 2 | CPU threads per worker |

### Architecture

```
verifiers_cross_validation.py
  ├── model_preloader.scan_and_preload_models()
  │     ├── Scan functions for spaCy/trankit/NLTK usage
  │     ├── Verify xlm-roberta-base is cached (no download)
  │     ├── Download trankit language models from HF mirror
  │     └── Warmup: test trankit.Pipeline in subprocess
  ├── WorkerPool.create()
  │     ├── Pre-fork N workers
  │     ├── Pre-import torch, trankit, transformers (COW)
  │     ├── Pre-load trankit.Pipeline per language
  │     └── Monkey-patch trankit.Pipeline → cached instance
  └── ThreadPoolExecutor → cross_validator.run_cross_validation()
        └── FunctionExecutor.test_function_batch()
              └── WorkerPool.run_function() or subprocess fallback
```

**Checkpoint support**: results are written incrementally to the output JSONL.
On SLURM restart, already-processed items are skipped automatically.

### NLP model dependencies

#### trankit

The official trankit model server has been offline since mid-2025. The model
preloader downloads from the HuggingFace mirror at
[uonlp/trankit](https://huggingface.co/uonlp/trankit/tree/main/models).
Models are cached in `cache/trankit/xlm-roberta-base/<language>/`.

The worker pool monkey-patches `trankit.Pipeline` to return cached instances,
avoiding the ~15s initialization per call that LLM-generated code would trigger.

#### spaCy

Models are pip packages installed via `python -m spacy download`. The preloader
checks if already installed before attempting download.

#### NLTK

Data (punkt, wordnet, etc.) is downloaded via `nltk.download()`.

### GPU cross-validation (removed)

A GPU-based cross-validation path (`cross_validation_gpu_job`) was attempted but
removed. The CPU worker pool approach proved sufficient (~593 items validated in
<90 min across checkpoint restarts). The GPU path was abandoned due to:

- Python 3.12 in the ROCm container breaking trankit's vendored `adapter_transformers`
- `huggingface-hub` version mismatches in the ROCm container
- Fragile `sed`-based patching of installed packages

## Concatenation job

The concatenation job (`src/concat_queries.py`) is a preprocessing step between cross-validation and response generation. It combines validated constraints (instructions) with user queries to produce a dataset that the response job later uses to dynamically build prompts.

The script:

1. Loads validated verifiers from the cross-validation output, keeping only successfully validated instructions and their passing eval functions. Test cases and validation metadata are dropped.
2. Loads user queries in standard messages format from a JSONL file or directory.
3. For each output sample, pairs a query with a multi-turn selection of instructions. Instructions accumulate across turns — turn N contains all instructions from turns 0..N.
4. The number of new instructions added per turn is configurable: either a fixed count or a weighted random distribution.

### Configuration

```yaml
concatenation_job:
  type: cpu_script
  template: "cpu_slurm.sh.j2"
  script: src/concat_queries.py
  args:
    verifiers-file: "verifiers_validated.jsonl"
    queries-path: "/path/to/queries"
    output-file: "concat_queries.jsonl"
    num-output-lines: 300000
    instructions-per-turn: "1,0.5,0.25"
    turns: 3
```

`--instructions-per-turn` accepts either a single integer (fixed count) or a comma-separated list of weights. With `"1,0.5,0.25"` each turn independently samples how many new instructions to add: 1 new instruction with relative weight 1.0, 2 with weight 0.5, or 3 with weight 0.25.

### Input: validated verifiers (`verifiers_validated.jsonl`)

Each line contains a validated instruction with its eval functions and test case metadata. The concatenation job only uses entries where `success` is `true` and keeps only `passing_functions` as the eval functions:

```json
{
  "instruction_id": "33",
  "instruction": "Your response must contain at least {N} words that are translators.",
  "instruction_category": "Content",
  "placeholders": {"N": {"type": "numeric"}},
  "eval_funcs": ["def evaluate(response, **kwargs): ..."],
  "cases": [{"input": {...}, "output": true}, "..."],
  "success": true,
  "passing_functions": ["def evaluate(response, **kwargs): ...", "..."],
  "passing_cases": ["..."],
  "best_accuracy": 1.0,
  "total_functions": 5,
  "total_cases": 32,
  "failure_reason": null
}
```

### Input: queries (messages format)

Standard chat messages JSONL. Any extra fields besides `messages` are preserved as `query_metadata`:

```json
{"messages": [{"role": "user", "content": "What is REST?"}], "sample_id": 134420}
```

### Output: concatenated samples (`concat_queries.jsonl`)

Each output line pairs a query with a multi-turn instruction selection. Instructions accumulate across turns. The `eval_funcs` per instruction is the list of cross-validated passing functions:

```json
{
  "messages": [{"role": "user", "content": "What is REST?"}],
  "query_metadata": {"sample_id": 134420},
  "turns": [
    {
      "instruction_ids": ["33"],
      "instructions": ["Your response must contain at least {N} words that are translators."],
      "instruction_categories": ["Content"],
      "placeholders": [{"N": {"type": "numeric"}}],
      "eval_funcs": [["def evaluate(response, **kwargs): ..."]]
    },
    {
      "instruction_ids": ["33", "78"],
      "instructions": [
        "Your response must contain at least {N} words that are translators.",
        "Your response must be written in exactly {N} paragraphs."
      ],
      "instruction_categories": ["Content", "Format"],
      "placeholders": [{"N": {"type": "numeric"}}, {"N": {"type": "numeric"}}],
      "eval_funcs": [
        ["def evaluate(response, **kwargs): ..."],
        ["def evaluate(response, **kwargs): ...", "def evaluate(response, **kwargs): ..."]
      ]
    },
    {
      "instruction_ids": ["33", "78", "112"],
      "instructions": ["...(accumulated from previous turns)...", "...", "...(new)..."],
      "instruction_categories": ["Content", "Format", "Length"],
      "placeholders": ["..."],
      "eval_funcs": [["..."], ["..."], ["..."]]
    }
  ],
  "source": "/path/to/queries"
}
```
