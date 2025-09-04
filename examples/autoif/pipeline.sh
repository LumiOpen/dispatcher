#!/bin/bash

# =============================================================================
# CORE PIPELINE CONFIGURATION
# =============================================================================

LANGUAGE="${LANGUAGE:-en}"
VENV_DIR="${VENV_DIR:-.venv}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-requirements.txt}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            echo "Usage: $0 [OPTIONS] [model_path]"
            echo "AutoIF Pipeline Script - Runs both Phase 1 (verifier generation) and Phase 2 (response generation & SFT dataset creation)"
            echo "Options:"
            echo "  --seed_file <path>       Path to seed instructions file"
            echo "  --queries_dataset <path> Path to queries dataset (HuggingFace path or local jsonl file)"
            echo "  --out_dir <path>         Path to experiment output directory"
            echo "  --help                   Show this help message"
            echo ""
            echo "Arguments:"
            echo "  model_path               Specify the model to use (default: 'meta-llama/Llama-3.3-70B-Instruct')"
            echo ""
            exit 0
            ;;
        --seed_file)
            SEED_FILE_ARG="$2"
            shift 2
            ;;
        --queries_dataset)
            QUERIES_DATASET_ARG="$2"
            shift 2
            ;;
        --out_dir)
            OUT_DIR_ARG="$2"
            shift 2
            ;;
        -*)
            echo "Error: Unknown option $1"
            echo "Use --help for usage information"
            exit 1
            ;;
        *)
            if [[ -z "$MODEL_SET" ]]; then
                MODEL_ARG="$1"
                MODEL_SET=true
                shift
            else
                echo "Error: Multiple model paths provided"
                echo "Use --help for usage information"
                exit 1
            fi
            ;;
    esac
done

# Read these from args, otherwise read from env vars, lastly fallback to defaults
MODEL="${MODEL_ARG:-${MODEL:-meta-llama/Llama-3.3-70B-Instruct}}"
export OUT_DIR="${OUT_DIR_ARG:-${OUT_DIR:-data/ifeval}}" # exported for verifiers_cross_validation.py to access
SEED_FILE="${SEED_FILE_ARG:-${SEED_FILE:-data/seed_instructions_ifeval.txt}}"
QUERIES_DATASET="${QUERIES_DATASET_ARG:-${QUERIES_DATASET:-/scratch/project_462000353/posttraining_data/lmsys-chat-1m/unredacted_filtered_dedup_eng.jsonl}}"

echo "Using model: $MODEL"
echo "Using experiment directory: $OUT_DIR"

# =============================================================================
# STEP SKIPPING CONFIGURATION
# =============================================================================
SKIP_AUGMENTATION="${SKIP_AUGMENTATION:-false}"
SKIP_VERIFIERS="${SKIP_VERIFIERS:-false}"
SKIP_CONCAT="${SKIP_CONCAT:-false}"
SKIP_RESPONSES="${SKIP_RESPONSES:-false}"
SKIP_SFT="${SKIP_SFT:-false}"

# =============================================================================
# LAUNCH SCRIPT CONFIGURATION (exported for use by SLURM launch scripts)
# =============================================================================

# Core settings exported to launch scripts
export MODEL
export LANGUAGE
export HF_HOME="${HF_HOME:-/scratch/project_462000353/hf_cache}"

# Intermediate files (exported for launch scripts)
export AUGMENT_INPUT_FILE="${AUGMENT_INPUT_FILE:-${OUT_DIR}/aug_input.jsonl}"
export AUGMENT_OUTPUT_FILE="${AUGMENT_OUTPUT_FILE:-${OUT_DIR}/aug_output.jsonl}"
export VERIFIERS_INPUT_FILE="${VERIFIERS_INPUT_FILE:-${OUT_DIR}/verifiers_input.jsonl}"
export VERIFIERS_OUTPUT_FILE="${VERIFIERS_OUTPUT_FILE:-${OUT_DIR}/verifiers_output.jsonl}"
export VERIFIERS_QUERIES_FILE="${VERIFIERS_QUERIES_FILE:-${OUT_DIR}/verifiers_queries.jsonl}"
export SCORED_RESPONSES_FILE="${SCORED_RESPONSES_FILE:-${OUT_DIR}/scored_responses.jsonl}"

# =============================================================================
# STEP 1: INSTRUCTION AUGMENTATION CONFIGURATION
# =============================================================================
# Input files

NUM_OF_AUGMENTED_INSTRUCTIONS="${NUM_OF_AUGMENTED_INSTRUCTIONS:-100}"
AUGMENTED_INSTRUCTIONS_FILE="${AUGMENTED_INSTRUCTIONS_FILE:-${OUT_DIR}/augmented_instructions.csv}"

# =============================================================================
# STEP 2: VERIFIER GENERATION CONFIGURATION  
# =============================================================================

VERIFIERS_ALL_FILE="${VERIFIERS_ALL_FILE:-${OUT_DIR}/verifiers_all.jsonl}"
VERIFIERS_FILTERED_FILE="${VERIFIERS_FILTERED_FILE:-${OUT_DIR}/verifiers_filtered.jsonl}"

# Verifier execution settings (exported for launch scripts)
export FUNCTION_TIMEOUT="${FUNCTION_TIMEOUT:-5}"
export MIN_FUNCTIONS="${MIN_FUNCTIONS:-1}"
export MIN_TEST_CASES="${MIN_TEST_CASES:-1}"
export ACCURACY_THRESHOLD="${ACCURACY_THRESHOLD:-0.8}"

# =============================================================================
# STEP 3: QUERIES CONCATENATION CONFIGURATION
# =============================================================================

# Query concatenation settings - use custom values if provided, otherwise use defaults
QUERY_COLUMN_NAME="${QUERY_COLUMN_NAME:-queries}"
RESPONSE_COLUMN_NAME="${RESPONSE_COLUMN_NAME:-responses}"
INSTRUCTIONS_PER_QUERY="${INSTRUCTIONS_PER_QUERY:-1}"
QUERY_MAX_LEN="${QUERY_MAX_LEN:-200}"
NUM_OUTPUT_LINES="${NUM_OUTPUT_LINES:-300000}"
MESSAGES_FORMAT="${MESSAGES_FORMAT:-true}"
MESSAGES_KEY="${MESSAGES_KEY:-messages}"
TURNS="${TURNS:-2}"
NO_FOLLOWUP="${NO_FOLLOWUP:-true}"

# =============================================================================
# STEP 4: RESPONSE GENERATION CONFIGURATION
# =============================================================================

# SFT dataset filtering
export SCORE_THRESHOLD="${SCORE_THRESHOLD:-4}"

# =============================================================================
# STEP 5: FINAL SFT
# =============================================================================

# final SFT dataset
SFT_DATASET_DIR="${SFT_DATASET_DIR:-${OUT_DIR}/sft_dataset}"

# Checkpointing mechanism
mkdir -p logs
CHECKPOINT_FILE="${CHECKPOINT_FILE:-${OUT_DIR}/state_tracker.log}"
touch "$CHECKPOINT_FILE"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# Get pipeline continuation point from the Python implementation
determine_continuation_point() {
    if [ ! -f "$CHECKPOINT_FILE" ]; then
        echo "AUG_START"
        return
    fi

    python src/utils/checkpoint.py --checkpoint_file "$CHECKPOINT_FILE" get-continuation
}

# Function to check if a step has been completed
step_completed() {
    local step_name="$1"
    python src/utils/checkpoint.py --checkpoint_file "$CHECKPOINT_FILE" check --step "$step_name"
    return $?
}

# Function to mark a step as completed
mark_step_completed() {
    local step_name="$1"
    python src/utils/checkpoint.py --checkpoint_file "$CHECKPOINT_FILE" mark --step "$step_name"
}

# Function to check if a step should be skipped
skip_step() {
    local step_name="$1"
    local skip_var="$2"
    
    if [[ "${!skip_var}" == "true" ]]; then
        echo "${skip_var}=true: Skipping ${step_name}"
        echo "${step_name} skipped - using existing files"
        return 0  # Step should be skipped
    fi
    return 1  # Step should be executed
}

# Function to execute a simple step with checkpoint management
execute_step() {
    local step_name="$1"
    local checkpoint="$2"
    local command="$3"
    local error_msg="$4"
    
    if ! step_completed "$checkpoint"; then
        echo "${step_name}"
        echo "Executing ${command}"
        eval "$command"
        if [ $? -ne 0 ]; then
            echo "$error_msg"
            exit 1
        fi
        mark_step_completed "$checkpoint"
        echo "${step_name} completed successfully."
    else
        echo "${step_name} already completed, skipping."
    fi
}

submit_slurm_job() {
    local launch_script=$1          # Path to the launch script
    local job_id_prefix=$2          # Prefix for the job ID in checkpoint file (e.g., "AUG_JOB_ID", "VER_JOB_ID")
    local submitted_step=$3         # Step name to mark as submitted
    local job_type_label=$4         # Label for logging (e.g., "inference", "verifier generation")
    
    echo "${job_id_prefix%%_*}: Submitting ${job_type_label} job"
    
    # Submit the job and capture the job ID
    local job_id=$(sbatch $launch_script | awk '{print $4}')

    echo "${job_type_label} job submitted with ID: $job_id"
    
    # Mark as submitted for checkpointing
    mark_step_completed $submitted_step
    
    # Save job ID to checkpoint file for later recovery
    python src/utils/checkpoint.py --checkpoint_file "$CHECKPOINT_FILE" save-job --prefix "$job_id_prefix" --job-id "$job_id"

    # set the job ID for further processing in the parent scope
    LAST_SUBMITTED_JOB_ID=$job_id
}

# Function to handle job submission with failure recovery
handle_job_submission() {
    local launch_script="$1"
    local job_id_prefix="$2"
    local submitted_step="$3"
    local failed_step="$4"
    local complete_step="$5"
    local job_type_label="$6"
    local job_var_name="$7"
    
    # Check if previous run failed and we need to resubmit
    if step_completed "$failed_step" && ! step_completed "$complete_step"; then
        echo "${job_id_prefix%%_*}: Previous inference job failed, resubmitting..."
        submit_slurm_job "$launch_script" "$job_id_prefix" "$submitted_step" "$job_type_label"
        eval "${job_var_name}=\$LAST_SUBMITTED_JOB_ID"
    elif ! step_completed "$submitted_step"; then
        submit_slurm_job "$launch_script" "$job_id_prefix" "$submitted_step" "$job_type_label"
        eval "${job_var_name}=\$LAST_SUBMITTED_JOB_ID"
    else
        echo "${job_id_prefix%%_*}: Inference job already submitted, retrieving job ID."
        local job_id=$(python src/utils/checkpoint.py --checkpoint_file "$CHECKPOINT_FILE" get-job --prefix "$job_id_prefix")
        echo "Retrieved ${job_type_label} Job ID: $job_id"
        eval "${job_var_name}=\$job_id"
    fi
}

# Function to monitor and verify job completion
monitor_and_verify_job() {
    local job_id="$1"
    local prefix="$2"
    local input_file="$3"
    local output_file="$4"
    local complete_step="$5"
    local failed_step="$6"
    local required_completion="$7"
    local job_type_label="$8"
    
    if ! step_completed "$complete_step"; then
        echo "${prefix}: Monitoring ${job_type_label} job"
        echo "Current job ID: $job_id"
        local log_file="logs/${job_id}_${prefix,,}.out"
        
        # First monitor the job status
        bash src/utils/monitor_slurm_job.sh \
            --job_id "$job_id" \
            --prefix "$prefix" \
            --check_interval 30
        
        if [ $? -ne 0 ]; then
            echo "${prefix}: Job failed. Check logs and restart script."
            mark_step_completed "$failed_step"
            exit 1
        fi
        
        # Once job is completed, verify the output files
        python src/utils/verify_job_output.py \
            --input_file "$input_file" \
            --output_file "$output_file" \
            --log_file "$log_file" \
            --required_completion "$required_completion" \
            --prefix "$prefix"
        
        if [ $? -ne 0 ]; then
            echo "${prefix}: Job output verification failed. Check logs and restart script."
            mark_step_completed "$failed_step"
            exit 1
        fi
        
        # Mark step as completed in checkpoint file
        mark_step_completed "$complete_step"
        echo "${prefix}: Inference job completed successfully!"
    else
        echo "${prefix}: Inference job already completed, skipping."
    fi
}

# Function to set up virtual environment if it doesn't exist
setup_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        echo "Creating virtual environment at $VENV_DIR"
        
        # Check if Python is installed
        if ! command -v python &> /dev/null; then
            echo "ERROR: Python is not installed!"
            exit 1
        fi
        
        # Create virtual environment
        python -m venv $VENV_DIR --system-site-packages
        
        # Activate the virtual environment
        source "$VENV_DIR/bin/activate"
        
        # Install dependencies
        pip install -r "$REQUIREMENTS_FILE"
        
        # Check if installation was successful
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to install dependencies!"
            deactivate
            exit 1
        fi

        deactivate
        echo "Virtual environment setup completed successfully."
    else
        echo "Virtual environment already exists at $VENV_DIR"
    fi
}

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

module use /appl/local/csc/modulefiles
module load pytorch/2.5
# Setup virtual environment
setup_venv
# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# =============================================================================
# PIPELINE EXECUTION
# =============================================================================

# Define step constants
AUG_PREPROCESSING="AUG_PREPROCESSING"
AUG_INFERENCE_SUBMITTED="AUG_INFERENCE_SUBMITTED"
AUG_INFERENCE_FAILED="AUG_INFERENCE_FAILED"
AUG_INFERENCE_COMPLETE="AUG_INFERENCE_COMPLETE"
AUG_POSTPROCESSING="AUG_POSTPROCESSING"

VER_PREPROCESSING="VER_PREPROCESSING"
VER_INFERENCE_SUBMITTED="VER_INFERENCE_SUBMITTED"
VER_INFERENCE_COMPLETE="VER_INFERENCE_COMPLETE"
VER_INFERENCE_FAILED="VER_INFERENCE_FAILED"
VER_CROSS_VALIDATION="VER_CROSS_VALIDATION"

CONCAT_QUERIES_CONCATED="CONCAT_QUERIES_CONCATED"
RESP_INFERENCE_SUBMITTED="RESP_INFERENCE_SUBMITTED"
RESP_INFERENCE_COMPLETE="RESP_INFERENCE_COMPLETE"
RESP_INFERENCE_FAILED="RESP_INFERENCE_FAILED"
SFT_DATASET_BUILT="SFT_DATASET_BUILT"

# Check and report checkpoint status
if [ ! -f "$CHECKPOINT_FILE" ]; then
    echo "No checkpoint, creating $CHECKPOINT_FILE"
else
    echo "Using checkpoint $CHECKPOINT_FILE"
fi

# Get the continuation point
CONTINUE_FROM=$(determine_continuation_point)
echo "Continuing pipeline execution from: $CONTINUE_FROM"

case $CONTINUE_FROM in
    "AUG_START")
        if ! skip_step "Instruction augmentation" "SKIP_AUGMENTATION"; then
            echo "Starting augmentation phase"
            echo "Using seed instructions file: $SEED_FILE"

            execute_step "AUG: Pre-processing instructions" "$AUG_PREPROCESSING" \
                "python src/create_instructions_input.py --seed_file $SEED_FILE --output_file $AUGMENT_INPUT_FILE --num_instructions $NUM_OF_AUGMENTED_INSTRUCTIONS" \
                "Pre-processing failed!"
        fi
        ;&  # Fallthrough
    "AUG_INFERENCE")
        if [[ "$SKIP_AUGMENTATION" != "true" ]]; then
            handle_job_submission "launch_augment_instructions.sh" "AUG_JOB_ID" \
                "$AUG_INFERENCE_SUBMITTED" "$AUG_INFERENCE_FAILED" "$AUG_INFERENCE_COMPLETE" \
                "instruction augmentation" "AUG_JOB_ID"
        fi
        ;&  # Fallthrough
    "AUG_MONITOR")
        if [[ "$SKIP_AUGMENTATION" != "true" ]]; then
            monitor_and_verify_job "$AUG_JOB_ID" "AUG" "$AUGMENT_INPUT_FILE" \
                "$AUGMENT_OUTPUT_FILE" "$AUG_INFERENCE_COMPLETE" "$AUG_INFERENCE_FAILED" \
                "100" "augmentation"
        fi
        ;&  # Fallthrough
    "AUG_POSTPROCESS")
        if [[ "$SKIP_AUGMENTATION" != "true" ]]; then
            execute_step "AUG: Post-processing inference results" "$AUG_POSTPROCESSING" \
                "python src/process_instructions_output.py --input_file $AUGMENT_OUTPUT_FILE --output_file $AUGMENTED_INSTRUCTIONS_FILE --seed_file $SEED_FILE --language $LANGUAGE --max_instructions $NUM_OF_AUGMENTED_INSTRUCTIONS" \
                "Post-processing failed!"
        fi
        ;&  # Fallthrough
    "VER_START")
        if ! skip_step "Verifier generation" "SKIP_VERIFIERS"; then
            echo "Starting verifier generation phase"
            
            execute_step "VER: Pre-processing instructions for verifier generation" "$VER_PREPROCESSING" \
                "python src/create_verifiers_input.py --instructions_file $AUGMENTED_INSTRUCTIONS_FILE --output_file $VERIFIERS_INPUT_FILE" \
                "Verifier pre-processing failed!"
        fi
        ;&  # Fallthrough
    "VER_INFERENCE")
        if [[ "$SKIP_VERIFIERS" != "true" ]]; then
            handle_job_submission "launch_generate_verifiers.sh" "VER_JOB_ID" \
                "$VER_INFERENCE_SUBMITTED" "$VER_INFERENCE_FAILED" "$VER_INFERENCE_COMPLETE" \
                "verifier generation" "VER_JOB_ID"
        fi
        ;&  # Fallthrough
    "VER_MONITOR")
        if [[ "$SKIP_VERIFIERS" != "true" ]]; then
            monitor_and_verify_job "$VER_JOB_ID" "VER" "$VERIFIERS_INPUT_FILE" \
                "$VERIFIERS_OUTPUT_FILE" "$VER_INFERENCE_COMPLETE" "$VER_INFERENCE_FAILED" \
                "90" "verifier generation"
        fi
        ;&  # Fallthrough
    "VER_CROSSVAL")
        if [[ "$SKIP_VERIFIERS" != "true" ]]; then
            execute_step "VER: Cross-validating and filtering verifiers" "$VER_CROSS_VALIDATION" \
                "python src/verifiers_cross_validation.py --verifiers_file $VERIFIERS_OUTPUT_FILE --output_all_file $VERIFIERS_ALL_FILE --output_filtered_file $VERIFIERS_FILTERED_FILE" \
                "Cross-validation failed!"
            echo "Phase 1 pipeline completed successfully!"
            echo "Generated verifiers are available at: $VERIFIERS_FILTERED_FILE"
        fi
        ;&  # Fallthrough
    "CONCAT_START")
        echo "Starting Phase 2: Concat queries with instructions, generate responses and build SFT dataset"
        if ! skip_step "Query concatenation" "SKIP_CONCAT"; then
            echo "Starting query concatenation phase"
            echo "Using queries dataset: $QUERIES_DATASET"
            
            # Prepare concat queries command with conditional arguments
            CONCAT_CMD="python src/concat_queries.py \
                --verifiers_file \"$VERIFIERS_FILTERED_FILE\" \
                --output_file \"$VERIFIERS_QUERIES_FILE\" \
                --queries_dataset \"$QUERIES_DATASET\" \
                --query_column_name \"$QUERY_COLUMN_NAME\" \
                --response_column_name \"$RESPONSE_COLUMN_NAME\" \
                --query_max_len \"$QUERY_MAX_LEN\" \
                --instructions_per_query \"$INSTRUCTIONS_PER_QUERY\" \
                --num_output_lines \"$NUM_OUTPUT_LINES\" \
                --turns \"$TURNS\" \
                --messages_key \"$MESSAGES_KEY\""
            
            [[ "$MESSAGES_FORMAT" == "true" ]] && CONCAT_CMD="$CONCAT_CMD --messages_format"
            [[ "$NO_FOLLOWUP" == "true" ]] && CONCAT_CMD="$CONCAT_CMD --no-followup"
            
            execute_step "CONCAT: Concat queries" "$CONCAT_QUERIES_CONCATED" \
                "$CONCAT_CMD" "Concat queries failed!"
        fi
        ;&  # Fallthrough
    "RESP_START")
        if ! skip_step "Response generation" "SKIP_RESPONSES"; then
            echo "Starting response generation phase"
            handle_job_submission "launch_generate_query_responses.sh" "RESP_JOB_ID" \
                "$RESP_INFERENCE_SUBMITTED" "$RESP_INFERENCE_FAILED" "$RESP_INFERENCE_COMPLETE" \
                "query response generation" "RESP_JOB_ID"
        fi
        ;&  # Fallthrough
    "RESP_MONITOR")
        if [[ "$SKIP_RESPONSES" != "true" ]]; then
            monitor_and_verify_job "$RESP_JOB_ID" "RESP" "$VERIFIERS_QUERIES_FILE" \
                "$SCORED_RESPONSES_FILE" "$RESP_INFERENCE_COMPLETE" "$RESP_INFERENCE_FAILED" \
                "90" "query response generation"
        fi
        ;&  # Fallthrough
    "SFT_START")
        if ! skip_step "SFT dataset building" "SKIP_SFT"; then
            echo "Starting SFT dataset building phase"
            mkdir -p "$SFT_DATASET_DIR"
            execute_step "SFT: Building SFT dataset from scored responses" "$SFT_DATASET_BUILT" \
                "python src/build_sft.py \"$SCORED_RESPONSES_FILE\" --output_dir \"$SFT_DATASET_DIR\" --score_threshold \"$SCORE_THRESHOLD\" --test" \
                "SFT dataset building failed!"
            echo "SFT: Dataset built successfully!"
        fi
        ;&  # Fallthrough
    "COMPLETE")
        echo "Pipeline completed successfully!"
        echo "Generated verifiers are available at: $VERIFIERS_FILTERED_FILE"
        echo "Query-instruction pairs with verifiers are available at: $VERIFIERS_QUERIES_FILE"
        echo "Scored responses are available at: $SCORED_RESPONSES_FILE"
        echo "SFT dataset is available at: $SFT_DATASET_DIR"
        ;;
esac

echo ""
echo "Pipeline completed successfully!"
echo "Generated verifiers are available at: $VERIFIERS_FILTERED_FILE"
echo "Query-instruction pairs with verifiers are available at: $VERIFIERS_QUERIES_FILE"
echo "Scored responses are available at: $SCORED_RESPONSES_FILE"
echo "SFT dataset is available at: $SFT_DATASET_DIR"

deactivate