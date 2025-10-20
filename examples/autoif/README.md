# AutoIF on Dispatcher

This repository implements [AutoIF (Automatic Instruction Following)](https://arxiv.org/abs/2402.04635) pipeline using the Dispatcher framework. AutoIF is a method that uses language models to verify instruction following and improve instruction tuning.

The implementation is divided into two phases:
1. **Phase 1**: Generate verification functions (verifiers) that can evaluate whether responses follow instructions. Steps in this phase are:
  - **AUGMENTATION**: augmenting the seed instructions (configure the input seed json with `--seed_file` argument)
  - **VERIFICATION**: generating and cross-validating the verifiers
2. **Phase 2**: Generate responses to queries, verify them with the functions, and create high-quality instruction-following data for fine-tuning. Steps in this phase are:
  - **CONCAT**: concatenating the verifiers with the queries (configure the queries dataset with `--queries_dataset` argument)
  - **RESPONSE**: generating responses for the concatenated queries
  - **SFT**: creating a supervised fine-tuning dataset from the responses

## Running the Complete Pipeline

The entire AutoIF pipeline (both Phase 1 and Phase 2) can be executed with a single command:

```bash
sh pipeline.sh [ --seed_file data/seed_instructions.txt ] [ --queries_dataset data/queries.jsonl ] [ --out_dir data/out1 ] [ model/path ]
```

The pipeline utilizes checkpointing mechanism (stored in `{out_dir}/state_tracker.log`) that tracks completion of each step, enabling safe restarts if any step fails or needs to be rerun.

You can also explicitly execute specific steps by setting the EXECUTE_{STEP} environment variables. If no steps are configured to be executed, the full pipeline will run by default.

For example, if you want to perform only the second phase of the pipeline (after verifiers have been generated), you can execute only the last three steps with

```bash
EXECUTE_CONCAT=true EXECUTE_RESPONSES=true EXECUTE_SFT=true sh pipeline.sh [ model/path ] \
[ --queries_dataset data/queries.jsonl ] \
[ --out_dir data/out1 ]
```

Alternatively, if you want to stop the pipeline after generating verifiers and before concatenating the queries from a dataset (a reason might be that you might not yet have the queries data or you might only be interested in the verifiers phase) then run

```bash
EXECUTE_AUGMENTATION=true EXECUTE_VERIFIERS=true sh pipeline.sh [ model/path ] \
[ --seed_file data/seed_instructions.txt ] \
[ --out_dir data/out1 ]
```

## Configuration

Steps can be configured by setting the respective env vars. Here is a complete list of configurable variables organized by steps with their defaults:

```sh
# core conf
LANGUAGE=en
VENV_DIR=.venv
REQUIREMENTS_FILE=requirements.txt
MODEL=meta-llama/Llama-3.3-70B-Instruct
OUT_DIR=data/ifeval # path to all data files. cmdline arg --out_dir will overwrite this
HF_HOME=/scratch/project_462000353/hf_cache

# step execution conf
EXECUTE_AUGMENTATION= # Set to 'true' to execute instruction augmentation step
EXECUTE_VERIFIERS=   # Set to 'true' to execute verifier generation step  
EXECUTE_CONCAT=      # Set to 'true' to execute query concatenation step
EXECUTE_RESPONSES=   # Set to 'true' to execute response generation step
EXECUTE_SFT=         # Set to 'true' to execute SFT dataset building step
# Note: If no EXECUTE_* variables are set, all steps will be executed by default

# instruction augmentation conf
SEED_FILE=data/seed_instructions_ifeval.txt # cmdline arg --seed_file will overwrite this
AUGMENT_INPUT_FILE=${OUT_DIR}/aug_input.jsonl
AUGMENT_OUTPUT_FILE=${OUT_DIR}/aug_output.jsonl
NUM_OF_AUGMENTED_INSTRUCTIONS_PER_CATEGORY=50
MAX_AUGMENTED_INSTRUCTIONS=200
AUGMENTED_INSTRUCTIONS_FILE=${OUT_DIR}/augmented_instructions.csv

# verifiers generation conf
VERIFIERS_INPUT_FILE=${OUT_DIR}/verifiers_input.jsonl
VERIFIERS_OUTPUT_FILE=${OUT_DIR}/verifiers_output.jsonl
VERIFIERS_ALL_FILE=${OUT_DIR}/verifiers_all.jsonl
VERIFIERS_FILTERED_FILE=${OUT_DIR}/verifiers_filtered.jsonl
# cross-validation params
FUNCTION_TIMEOUT=10
MIN_FUNCTIONS=1
MIN_TEST_CASES=1
ACCURACY_THRESHOLD=0.8

# queries-instructions concatenation conf
VERIFIERS_QUERIES_FILE=${OUT_DIR}/verifiers_queries.jsonl # the output of this step
QUERIES_DATASET=/scratch/project_462000353/posttraining_data/lmsys-chat-1m/unredacted_filtered_dedup_eng.jsonl # cmdline arg --queries_dataset will overwrite this
QUERY_COLUMN_NAME=queries # used only if MESSAGES_FORMAT=false
RESPONSE_COLUMN_NAME==responses # used only if MESSAGES_FORMAT=false
INSTRUCTIONS_PER_QUERY=1
QUERY_MAX_LEN=200
NUM_OUTPUT_LINES=300000
MESSAGES_FORMAT=true
MESSAGES_KEY=messages
TURNS=1
NO_FOLLOWUP=true
BALANCE_CATEGORIES=true # if true, try to balance the number of queries per instruction category

# response generation conf
SCORED_RESPONSES_FILE=${OUT_DIR}/scored_responses.jsonl
SCORE_THRESHOLD=4 # Here used for intermediate judgements in multi-turn generations (scale from 1-5). Also used in next step building sft data

# build sft conf
SFT_DATASET_DIR=${OUT_DIR}/sft_dataset
```

## Examples

### IFEval-based constraints + lmsys-chat queries two-turn two-constraints each turn (cumulative constraints)

```sh
SEED_FILE=data/seed_instructions_ifeval.txt \
NUM_OF_AUGMENTED_INSTRUCTIONS=100 \
TURNS=2 \
INSTRUCTIONS_PER_QUERY=2 \
QUERY_MAX_LEN=500 \
NUM_OUTPUT_LINES=300000 \
OUT_DIR=data/ifeval-lmsys-2turn-2constraint \
QUERIES_DATASET=/scratch/project_462000353/posttraining_data/lmsys-chat-1m/unredacted_filtered_dedup_eng.jsonl \
sh pipeline.sh /scratch/project_462000353/zosaelai2/models/Llama-3.3-70B-Instruct

# if you already have the verifiers pre-generated, copy it to the out_dir and execute only the last three steps
cp path/to/pregenerated/verifiers.jsonl data/ifeval-lmsys-2turn-2constraint/verifiers_filtered.jsonl

EXECUTE_CONCAT=true \
EXECUTE_RESPONSES=true \
EXECUTE_SFT=true \
VERIFIERS_FILTERED_FILE=data/ifeval-lmsys-2turn-2constraint/verifiers_filtered.jsonl \ # optional, pipeline will discover automatically
TURNS=2 \
INSTRUCTIONS_PER_QUERY=2 \
QUERY_MAX_LEN=500 \
NUM_OUTPUT_LINES=300000 \
OUT_DIR=data/ifeval-lmsys-2turn-2constraint \
QUERIES_DATASET=/scratch/project_462000353/posttraining_data/lmsys-chat-1m/unredacted_filtered_dedup_eng.jsonl \
sh pipeline.sh /scratch/project_462000353/zosaelai2/models/Llama-3.3-70B-Instruct
```

## Pipeline Structure

### Step 1: Augment Instructions

This step takes seed instructions and generates additional similar instructions:

1. **Pre-processing**: 
   - Creates input for the instruction augmentation
   - `src/create_instructions_input.py` transforms seed instructions from `SEED_FILE` into prompt format (`AUGMENT_INPUT_FILE`)

2. **Inference**: 
   - Submits instruction augmentation job to the dispatcher
   - Input: `AUGMENT_INPUT_FILE`
   - Output: `AUGMENT_OUTPUT_FILE`

3. **Post-processing**: 
   - `src/process_instructions_output.py` extracts and filters augmented instructions
   - Filters out non-target language samples and potential duplicates
   - Output: Filtered augmented instructions `AUGMENTED_INSTRUCTIONS_FILE`

### Step 2: Generate Verifiers

This step generates Python verification functions to evaluate instruction following:

1. **Pre-processing**:
   - `src/create_verifiers_input.py` transforms instructions into prompts for verifier generation
   - Input: `AUGMENTED_INSTRUCTIONS_FILE`
   - Output: `VERIFIERS_INPUT_FILE`

2. **Inference**:
   - Submits verifier generation jobs to the dispatcher
   - Generates 10 verification functions per instruction and 3 test cases per each function
   - Output: Raw verification functions and test cases `VERIFIERS_OUTPUT_FILE`

3. **Post-processing**:
   - Cross-validates verifier functions using `src/verifiers_cross_validation.py`
      1. Parse functions and test cases from LLM responses
      2. Validate function safety (no harmful code patterns)
      3. Deduplicate test cases
      4. Filter test cases that pass at least `MIN_FUNCTIONS` functions
      5. Keep only functions that meet `ACCURACY_THRESHOLD` (how many test cases they pass)
      6. Output results for further processing
   - Output: Filtered verification functions `VERIFIERS_FILTERED_FILE` (also post-processed verifiers `VERIFIERS_ALL_FILE`)


### Output Format

The final output of Phase 1 includes:
```json
{
    "instructions": ["Follow this specific instruction..."],
    "queries": ["User's query to answer"],
    "eval_func": [["def evaluate(response): ..."],..],
    "cases": [[{"input": "test case", "output": true}],..],
    "prompts": ["Please answer the query strictly following the instruction..."]
}
```

## Phase 2: Generating Responses and Scoring

Phase 2 is implemented as a [`GeneratorTask`](src/autoif_generator_task.py) utilizing the multi-step inference functionality of the dispatcher.
It can be run standalone with

```sh
sbatch launch_generate_query_responses.sh
```

### Process Steps

1. **Query Response Generation**:
   - Generate responses to queries following specific instructions
   - Input: `VERIFIERS_QUERIES_FILE`

2. **Post-processing**:
   - Verify responses using the verification functions
   - Filter out responses that don't pass verification

3. **Response Scoring**:
   - Use another LLM generation to score each verified response for relevance and quality
   - Append the score to the final result
   - For multi-turn setups also filter after each turn based on `SCORE_THRESHOLD`
   - Output: `SCORED_RESPONSES_FILE`

4. **Final Output**:
   - SFT format data, filtered by `SCORE_THRESHOLD`
   - Output: `SFT_DATASET_DIR`

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
