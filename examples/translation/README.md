# Finnish reasoning data generation

These are scripts for generating and evaluating Finnish reasoning prompt translations. Note: our current reasoning trace generation scripts are not included in this branch, but in the translate branch instead!

## Evaluating translation quality

We attempt evaluating the translation quality via a proxy task of solving the translated problems with a separate (fixed) reasoning model. It seems there are no statistically significant difference between the tested translation models that we could measure with the current approach.

There are 3 steps to run these evaluations:
1. Translate prompts with `launch_translate_task_sing.sh` with all candidate translation models.
2. Generate answers for each translation model with `launch_answer_task_sing.sh`.
3. Evaluate models with `evaluate_prompts.py` or check statistically significant differences with `significance.py`. Both of these require gold standard answers in addition to the generations as the input.


## Finnish reasoning traces quality evaluation

1. Translate prompts (deepscaler) with one model - this should be fixed in this experiment (for now deepseekv3)
    - `/scratch/project_462000353/adamhrin/dispatcher/examples/translation/data/default-train-sample-100_translations_DeepSeek-V3_fi.jsonl`
    - `/scratch/project_462000353/adamhrin/dispatcher/examples/translation/data/default-train-sample-100_translations_Qwen2.5-72B-Instruct_fi.jsonl`
2. Use different candidate models (non-reasoning) to generate reasoning traces + answer (although that's not interesting to us - perhaps can be affected by the prompt but in this stage we let the model generate the answer as well). The input to the model is the prompt guiding the model to generate reasoning traces + the actual math question in finnish

```sh
sbatch launch_answer_task.sh
```

3. Take a reasoning model (fixed) to get the answer given the translated prompt and generated reasoning trace. This can be one of deepseekv3, r1 or qwen3 (MoE)
4. The accuracy of the problem solving across the models in (2) is a proxy for how well the models in (2) translated the traces.