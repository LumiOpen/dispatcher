# Finnish reasoning data generation

These are scripts for generating and evaluating Finnish reasoning prompt translations. Note: our current reasoning trace generation scripts are not included in this branch, but in the translate branch instead!

## Evaluating translation quality

We attempt evaluating the translation quality via a proxy task of solving the translated problems with a separate (fixed) reasoning model. It seems there are no statistically significant difference between the tested translation models that we could measure with the current approach.

There are 3 steps to run these evaluations:
1. Translate prompts with `launch_translate_task_sing.sh` with all candidate translation models.
2. Generate answers for each translation model with `launch_answer_task_sing.sh`.
3. Evaluate models with `evaluate_prompts.py` or check statistically significant differences with `significance.py`. Both of these require gold standard answers in addition to the generations as the input.
