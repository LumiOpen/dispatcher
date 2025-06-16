# AutoIF on Dispatcher

This repository implements [AutoIF (Automatic Instruction Following)](https://arxiv.org/abs/2402.04635) pipeline using the Dispatcher framework. AutoIF is a method that uses language models to verify instruction following and improve instruction tuning.

The implementation is divided into two phases:
1. **Phase 1**: Generate verification functions (verifiers) that can evaluate whether responses follow instructions
2. **Phase 2**: Generate responses to queries, verify them with the functions, and create high-quality instruction-following data for fine-tuning

## Phase 1: Generating Verifiers

Phase 1 is implemented as a multi-step process with pre/post-processing steps and inference using the dispatcher. The pipeline utilizes a checkpointing mechanism (stored in `logs/phase1_checkpoint.log`) to enable restart and development.

### Entry Point

Run this script if you want to trigger execution of the full phase1. The checkpointing file `logs/phase1_checkpoint.log` will keep track of which step has already been executed and has logic to wait for the generation task to complete or restart if it failed.

```bash
sbatch phase1_pipeline.sh
```

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
   - Keeps only functions that pass validation criteria (accuracy > 0.8)
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

> **Note:** This is a prototype that might not function yet. The flow has not been tested and most likely requires some more implementation (data format handling, etc.)

Phase 2 is implemented as a [`GeneratorTask`](src/autoif_generator_task.py) utilizing the multi-step inference functionality of the dispatcher.

The launch script is `src/launch_scripts/launch_generate_query_responses.sh`

### Process Steps

1. **Query Response Generation**:
   - Generate responses to queries following specific instructions
   - Input: `data/verifiers_queries.jsonl`

2. **Post-processing**:
   - Verify responses using the verification functions
   - Filter out responses that don't pass verification

3. **Response Scoring**:
   - Use another LLM generation to score each verified response for relevance and quality
   - Filter out low-scoring responses
   - Format final dataset for supervised fine-tuning
   - Output: `data/filtered_responses.jsonl`


## Requirements

- Dispatcher framework
- Python 3.8+
- Required Python packages in `requirements.txt`
- Access to LUMI