# AutoIF pipeline

## Quick start

Copy `config.default.yaml` into an experiment directory under `data/` and edit it to set your parameters

```sh
mkdir -p data/your_experiment_name
cp examples/autoif/config.default.yaml data/your_experiment_name/config.yaml
```

Run the pipeline

```sh
sh pipeline.sh --out-dir data/your_experiment_name
```

This is a wrapper for `pipeline.py` which submits the jobs in order and takes care of re-running failed jobs if necessary. Jobs are submitted with dependencies, so that next job starts only after the previous one is completed.

The order of the jobs is configured in config.yaml under `pipeline`. Set true/false to enable/disable each job.:

```yaml
pipeline:
  - augmentation_generation: true
  - augmentation_postprocessing: true
  - verifiers_generation: true
  - cross_validation_job: true
  - concatenation_job: true
  - responses_generation: true
  - final_dataset_job: true
```

**Using pipeline flags**
```sh
sh pipeline.sh --out-dir data/your_experiment_name --force # rerun all steps from the beginning.
sh pipeline.sh --out-dir data/your_experiment_name --rerun-failed # restart failed jobs and all subsequent jobs (cancel pending jobs if exists).
sh pipeline.sh --out-dir data/your_experiment_name --continue # continue from the last failed job without rerunning it. Useful if partial result is enough
```

**Jobs** are configured in `jobs/`. These can be GPU jobs or CPU jobs (pre/postprocessing of data). The jobs read configuration from `config.yaml`.

**Tasks** jobs that use LLM generations are implemented as instances of dispatcher.taskmanager.task.base.GeneratorTask. The implementation of each task is in `tasks/`

## Running with vllm server in local file mode

To speed up experimentation, you can also run jobs with dispatcher in local file mode. This is especially useful if vllm server is running on the background and you want to connect to it from multiple jobs.

1. To launch the openai compatible vllm server directly, follow the [steps 0 and 1](../README.md#development) in Development section in the examples README.

2. Configure the endpoint in `config.yaml`:

```yaml
vllm_server:
  host: "127.0.0.1"
  port: 8000
```

3. Implement local versions of the jobs that use LLM generations implemented as tasks. Name them as `<original_task_name>_local.sh`. For example, you can create a new job file `jobs/autoif/responses_generation_local.py` from `jobs/autoif/responses_generation.py`. This job starts the dispatcher server and executes the task in local file mode.

4. Run the pipeline as usual:

```sh
sh pipeline.sh --out-dir data/your_experiment_name
```
