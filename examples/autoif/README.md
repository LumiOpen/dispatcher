# AutoIF on Dispatcher

This repository implements [AutoIF (Automatic Instruction Following)](https://arxiv.org/abs/2402.04635) pipeline using the Dispatcher framework. AutoIF is a method that uses language models to verify instruction following and improve instruction tuning.

The implementation is divided into two phases:
1. **Phase 1**: Generate verification functions (verifiers) that can evaluate whether responses follow instructions
2. **Phase 2**: Generate responses to queries, verify them with the functions, and create high-quality instruction-following data for fine-tuning

## Running the Complete Pipeline

The entire AutoIF pipeline (both Phase 1 and Phase 2) can be executed with a single command:

```bash
sh pipeline.sh [ model/path ]
```
Model defaults to `meta-llama/Llama-3.3-70B-Instruct`.

The pipeline utilizes a robust checkpointing mechanism (stored in `logs/state_tracker.log`) that tracks completion of each step, enabling safe restarts if any step fails or needs to be rerun.

## Pipeline Structure

### Phase 1: Generating Verifiers

Run with ```sh phase1_pipeline.sh```

Phase 1 is implemented as a multi-step process with pre/post-processing steps and inference using the dispatcher.

### Step 1: Augment Instructions

This step takes seed instructions and generates additional similar instructions:

1. **Pre-processing**: 
   - Creates input for the instruction augmentation
   - `src/create_instructions_input.py` transforms seed instructions into prompt format

2. **Inference**: 
   - Submits instruction augmentation job to the dispatcher
   - Input: Seed instructions (`data/seed_instructions_fi.txt`)
   - Output: Raw model generations (`data/tmp_aug_output.jsonl`)

3. **Post-processing**: 
   - `src/process_instructions_output.py` extracts and filters augmented instructions
   - Filters out non-target language samples and potential duplicates
   - Output: Filtered augmented instructions (`data/augmented_instructions.txt`)

### Step 2: Generate Verifiers

This step generates Python verification functions to evaluate instruction following:

1. **Pre-processing**:
   - `src/create_verifiers_input.py` transforms instructions into prompts for verifier generation
   - Input: Augmented instructions (`data/augmented_instructions.txt`)
   - Output: Verifier generation prompts (`data/verifiers_input.jsonl`)

2. **Inference**:
   - Submits verifier generation jobs to the dispatcher
   - Generates multiple verification functions per instruction
   - Output: Raw verification functions and test cases (`data/verifiers_output.jsonl`)

3. **Post-processing**:
   - Cross-validates verifier functions using `src/verifiers_cross_validation.py`
      1. Parse functions and test cases from LLM responses
      2. Validate function safety (no harmful code patterns)
      3. Deduplicate test cases
      4. Filter test cases that pass at least MIN_FUNCTIONS functions
      5. Keep only functions that meet ACCURACY_THRESHOLD (how many test cases they pass)
      6. Output results for further processing
   - Output: Filtered verification functions (`data/filtered_verifiers.jsonl`)

4. **Query Augmentation**:
   - Combines verifiers with queries using `src/concat_queries.py`
   - Each instruction is paired with multiple queries
   - Output: Query-instruction pairs (`data/verifiers_queries.jsonl`)

### Output Format

The final output of Phase 1 includes:
```json
{
    "instruction": "Follow this specific instruction...",
    "query": "User's query to answer",
    "eval_func": ["def evaluate(response): ..."],
    "cases": [{"input": "test case", "output": true}],
    "prompt": "Please answer the query strictly following the instruction..."
}
```

## Phase 2: Generating Responses and Scoring

Phase 2 is implemented as a [`GeneratorTask`](src/autoif_generator_task.py) utilizing the multi-step inference functionality of the dispatcher.

```sh
cd src
sbatch launch_generate_query_responses.sh
```

### Process Steps

1. **Query Response Generation**:
   - Generate responses to queries following specific instructions
   - Input: `data/verifiers_queries.jsonl`

2. **Post-processing**:
   - Verify responses using the verification functions
   - Filter out responses that don't pass verification

3. **Response Scoring**:
   - Use another LLM generation to score each verified response for relevance and quality
   - Append the score to the final result
   - Output: `data/scored_responses.jsonl`

4. **Final Output**:
   - Run `python3 src/build_sft.py data/scored_responses.jsonl --output data/final_sft.jsonl --score_threshold 8` to build the final output for fine-tuning, scored above 8 out of 10 by an LLM judge.

## Requirements

- Dispatcher framework
- Python 3.8+
- Required Python packages in `requirements.txt`
- Access to LUMI


## Utilites

### Verifiers code preview 

The main use for this utility is to display the python code of the verification functions in human-readable format. The usage is

```sh
python src/utils/preview_verifiers.py <path/to/verifiers/file.jsonl> --instruction_id <id> [ --max_func <max_func> ] [ --tofile ]
```

<path/to/verifiers/file.jsonl> can be either the raw verifiers output file from dispatcher or the files resulting from cross-validation (filtered/all verifiers). However, note that the processed files do not  The `--instruction_id` is the id of the instruction to preview.

This will display all or `--max_func` python functions from the "eval_func" list, along with usage examples in "cases" list. If `--max_func` not supplied, print all.

Each evaluation function is a string with a python code. This utility will format the code and print it out in the console. 

Each test case is a json object with keys "input" and "output". Input is the input to the function and output is the expected output. These are printed at the end of the console output.

If flag `--tofile` is set, output is printed into a python file where each evaluation function is ordered with numbers like evaluate_1(), evaluate_2(), etc. At the end of the file is example usage formed out of the test cases. The python filename is `path/to/verifiers/file_<instruction_id>.py`
