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
