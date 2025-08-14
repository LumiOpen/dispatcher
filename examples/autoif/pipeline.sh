#!/bin/bash

# AutoIF Pipeline Script
RUNID="ifeval"

# =============================================================================
# CORE PIPELINE CONFIGURATION
# =============================================================================

# Default model and language settings
LANGUAGE="en"
MODEL="meta-llama/Llama-3.3-70B-Instruct"  # Default model
SEED_FILE_DEFAULT="data/seed_instructions_${RUNID}.txt"

# Virtual environment settings
VENV_DIR=".venv"
REQUIREMENTS_FILE="requirements.txt"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            echo "Usage: $0 [OPTIONS] [model_path]"
            echo "AutoIF Pipeline Script - Runs both Phase 1 (verifier generation) and Phase 2 (response generation & SFT dataset creation)"
            echo "Options:"
            echo "  --seed <path>           Path to seed instructions file"
            echo "  --help                  Show this help message"
            echo ""
            echo "Arguments:"
            echo "  model_path              Specify the model to use (default: 'meta-llama/Llama-3.3-70B-Instruct')"
            echo ""
            echo "Environment Variables for Step Skipping:"
            echo "  SKIP_AUGMENTATION=true     Skip instruction augmentation step"
            echo "  SKIP_VERIFIERS=true        Skip verifier generation step"
            echo "  SKIP_CONCAT=true           Skip query concatenation step"
            echo "  SKIP_RESPONSES=true        Skip query response generation step"
            exit 0
            ;;
        --seed)
            SEED_FILE_CUSTOM="$2"
            shift 2
            ;;
        -*)
            echo "Error: Unknown option $1"
            echo "Use --help for usage information"
            exit 1
            ;;
        *)
            if [[ -z "$MODEL_SET" ]]; then
                MODEL="$1"
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

echo "Using model: $MODEL"

# =============================================================================
# STEP SKIPPING CONFIGURATION
# =============================================================================
# Set to 'true' to skip specific pipeline steps
SKIP_AUGMENTATION=${SKIP_AUGMENTATION:-false}
SKIP_VERIFIERS=${SKIP_VERIFIERS:-false}
SKIP_CONCAT=${SKIP_CONCAT:-false}
SKIP_RESPONSES=${SKIP_RESPONSES:-false}

# =============================================================================
# DATA FILE PATHS (organized by pipeline phase)
# =============================================================================

# Input files
SEED_FILE="${SEED_FILE_CUSTOM:-$SEED_FILE_DEFAULT}"
echo "Using seed instructions file: $SEED_FILE"
QUERIES_DATASET="data/queries.jsonl" # can be a hf-like dataset or jsonl file

# Step 1: Instruction Augmentation files
AUGMENTED_INSTRUCTIONS_FILE="data/augmented_instructions_${RUNID}.csv"

# Step 2: Verifier Generation files
VERIFIERS_ALL_FILE="data/verifiers_all_${RUNID}.jsonl"
VERIFIERS_FILTERED_FILE="data/verifiers_filtered_${RUNID}.jsonl"

# Final output files
SFT_DATASET_FILE="data/sft_dataset_${RUNID}.jsonl"

# =============================================================================
# LAUNCH SCRIPT CONFIGURATION (exported for use by SLURM launch scripts)
# =============================================================================

# Core settings exported to launch scripts
export MODEL
export LANGUAGE
export HF_HOME="/scratch/project_462000353/hf_cache"

# Intermediate files (exported for launch scripts)
export AUGMENT_INPUT_FILE="data/aug_input_${RUNID}.jsonl"
export AUGMENT_OUTPUT_FILE="data/aug_output_${RUNID}.jsonl"
export VERIFIERS_INPUT_FILE="data/verifiers_input_${RUNID}.jsonl" 
export VERIFIERS_OUTPUT_FILE="data/verifiers_output_${RUNID}.jsonl"
export VERIFIERS_QUERIES_FILE="data/verifiers_queries_${RUNID}.jsonl"
export SCORED_RESPONSES_FILE="data/scored_responses_${RUNID}.jsonl"

# =============================================================================
# STEP 1: INSTRUCTION AUGMENTATION CONFIGURATION
# =============================================================================
NUM_OF_AUGMENTED_INSTRUCTIONS=100

# =============================================================================
# STEP 2: VERIFIER GENERATION CONFIGURATION  
# =============================================================================

# Verifier execution settings (exported for launch scripts)
export FUNCTION_TIMEOUT=5  # seconds - timeout for verifier function execution
export MIN_FUNCTIONS=1     # minimum number of functions required (original autoif: 3)
export MIN_TEST_CASES=1    # minimum number of test cases required (original autoif: 10)
export ACCURACY_THRESHOLD=0.8  # minimum accuracy threshold for validation

# Query concatenation settings
CONCAT_NUM_OUTPUT_LINES=${CONCAT_NUM_OUTPUT_LINES:-} # default: unset, uses script default
CONCAT_MESSAGES_FORMAT=${CONCAT_MESSAGES_FORMAT:-}   # default: unset, uses script default
CONCAT_MESSAGES_KEY=${CONCAT_MESSAGES_KEY:-}         # default: unset, uses script default
CONCAT_TURNS=${CONCAT_TURNS:-}                       # default: unset, uses script default
CONCAT_NO_FOLLOWUP=${CONCAT_NO_FOLLOWUP:-false}      # set to 'true' to disable follow-up generation

# =============================================================================
# STEP 3: RESPONSE GENERATION CONFIGURATION
# =============================================================================

# Response generation performance settings (exported for launch scripts)
export RESP_WORKERS=${RESP_WORKERS:-32}                    # simultaneous backend requests
export RESP_REQUEST_TIMEOUT=${RESP_REQUEST_TIMEOUT:-600}   # timeout for individual requests (seconds)
export RESP_WORK_TIMEOUT=${RESP_WORK_TIMEOUT:-1800}        # timeout for work items (seconds)

# SFT dataset filtering
SCORE_THRESHOLD=${SCORE_THRESHOLD:-4}  # minimum score for responses to be included in SFT dataset

# Checkpointing mechanism
mkdir -p logs
CHECKPOINT_FILE="logs/state_tracker_${RUNID}.log"
touch "$CHECKPOINT_FILE"

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

module use /appl/local/csc/modulefiles
module load pytorch/2.5
# Setup virtual environment
setup_venv
# Activate the virtual environment
source "$VENV_DIR/bin/activate"

echo "Starting AutoIF Pipeline: Phase 1 - Generating verifiers, Phase 2 - Generating responses and building SFT dataset"

# Step 1: Augment instructions
AUG_PREPROCESSING="AUG_PREPROCESSING"
AUG_INFERENCE_SUBMITTED="AUG_INFERENCE_SUBMITTED"
AUG_INFERENCE_FAILED="AUG_INFERENCE_FAILED"
AUG_INFERENCE_COMPLETE="AUG_INFERENCE_COMPLETE"
AUG_POSTPROCESSING="AUG_POSTPROCESSING"

# Check if augmentation step should be skipped
if [[ "$SKIP_AUGMENTATION" == "true" ]]; then
    echo "SKIP_AUGMENTATION=true: Skipping Step 1 (instruction augmentation) and marking all Step 1 checkpoints as complete"
    
    # Mark all Step 1 checkpoints as completed
    mark_step_completed "$AUG_PREPROCESSING"
    mark_step_completed "$AUG_INFERENCE_SUBMITTED"
    mark_step_completed "$AUG_INFERENCE_COMPLETE"
    mark_step_completed "$AUG_POSTPROCESSING"
    
    echo "Step 1 skipped - using existing augmented instructions file"
else
    echo "Starting Step 1: Augment instructions"
fi

# Only execute Step 1 if not skipped
if [[ "$SKIP_AUGMENTATION" != "true" ]]; then

    # Step 1: Pre-process - Create instructions input
    if ! step_completed $AUG_PREPROCESSING; then
        echo "AUG: Pre-processing instructions"
        python src/create_instructions_input.py \
            --seed_file $SEED_FILE \
            --output_file $AUGMENT_INPUT_FILE \
            --num_instructions $NUM_OF_AUGMENTED_INSTRUCTIONS
        if [ $? -ne 0 ]; then
            echo "Pre-processing failed!"
            exit 1
        fi
        echo "Pre-processing completed successfully."
        mark_step_completed $AUG_PREPROCESSING
    else
        echo "AUG: Pre-processing already completed, skipping."
    fi

    # Step 1.2: Submit inference job
    if step_completed $AUG_INFERENCE_FAILED; then
        echo "AUG: Previous inference job failed, resubmitting..."
        submit_slurm_job "launch_augment_instructions.sh" "AUG_JOB_ID" "$AUG_INFERENCE_SUBMITTED" "instruction augmentation"
        AUG_JOB_ID=$LAST_SUBMITTED_JOB_ID
    elif ! step_completed $AUG_INFERENCE_SUBMITTED; then
        submit_slurm_job "launch_augment_instructions.sh" "AUG_JOB_ID" "$AUG_INFERENCE_SUBMITTED" "instruction augmentation"
        AUG_JOB_ID=$LAST_SUBMITTED_JOB_ID
    else
        echo "AUG: Inference job already submitted, retrieving job ID."
        AUG_JOB_ID=$(python src/utils/checkpoint.py --checkpoint_file "$CHECKPOINT_FILE" get-job --prefix "AUG_JOB_ID")
        echo "Retrieved Augmentation Job ID: $AUG_JOB_ID"
    fi

    # Step 1.3: Wait for inference job to complete
    if ! step_completed $AUG_INFERENCE_COMPLETE; then
        echo "AUG: Waiting for inference job to complete"
        LOG_FILE="logs/${AUG_JOB_ID}_augment.out"
        
        # First monitor the job status
        bash src/utils/monitor_slurm_job.sh \
            --job_id "$AUG_JOB_ID" \
            --prefix "AUG" \
            --check_interval 30
        
        if [ $? -ne 0 ]; then
            echo "AUG: Job failed. Check logs and restart script."
            mark_step_completed $AUG_INFERENCE_FAILED
            exit 1
        fi
        
        # Once job is completed, verify the output files
        python src/utils/verify_job_output.py \
            --input_file "$AUGMENT_INPUT_FILE" \
            --output_file "$AUGMENT_OUTPUT_FILE" \
            --log_file "$LOG_FILE" \
            --required_completion 100 \
            --prefix "AUG"
        
        if [ $? -ne 0 ]; then
            echo "AUG: Job output verification failed. Check logs and restart script."
            mark_step_completed "$AUG_INFERENCE_FAILED"
            exit 1
        fi
        
        # Mark step as completed in checkpoint file
        mark_step_completed "$AUG_INFERENCE_COMPLETE"

        echo "AUG: Inference job completed successfully!"
    else
        echo "AUG: Inference job already completed, skipping."
    fi

    # Step 1.4: Post-process
    if ! step_completed $AUG_POSTPROCESSING; then
        echo "AUG: Post-processing inference results"
        python src/process_instructions_output.py \
            --input_file $AUGMENT_OUTPUT_FILE \
            --output_file $AUGMENTED_INSTRUCTIONS_FILE \
            --seed_file $SEED_FILE \
            --language $LANGUAGE \
            --max_instructions $NUM_OF_AUGMENTED_INSTRUCTIONS
        if [ $? -ne 0 ]; then
            echo "Post-processing failed!"
            exit 1
        fi
        mark_step_completed "$AUG_POSTPROCESSING"
    else
        echo "AUG: Post-processing already completed, skipping."
    fi

fi # End of Step 1 conditional execution

# Step 2: Generate verifiers
VER_PREPROCESSING="VER_PREPROCESSING"
VER_INFERENCE_SUBMITTED="VER_INFERENCE_SUBMITTED"
VER_INFERENCE_COMPLETE="VER_INFERENCE_COMPLETE"
VER_INFERENCE_FAILED="VER_INFERENCE_FAILED"
VER_CROSS_VALIDATION="VER_CROSS_VALIDATION"
VER_QUERIES_CONCATED="VER_QUERIES_CONCATED"

# Check if verifier generation step should be skipped
if [[ "$SKIP_VERIFIERS" == "true" ]]; then
    echo "SKIP_VERIFIERS=true: Skipping Step 2 (verifier generation) and marking all Step 2 checkpoints as complete"
    
    # Mark all Step 2 checkpoints as completed
    mark_step_completed "$VER_PREPROCESSING"
    mark_step_completed "$VER_INFERENCE_SUBMITTED"
    mark_step_completed "$VER_INFERENCE_COMPLETE"
    mark_step_completed "$VER_CROSS_VALIDATION"
    
    echo "Step 2 skipped - using existing verifier files"
else
    echo "Starting Step 2: Generate verifiers"
fi

# Only execute Step 2 if not skipped
if [[ "$SKIP_VERIFIERS" != "true" ]]; then

    # Step 2.1: Create verifiers input
    if ! step_completed $VER_PREPROCESSING; then
        echo "VER: Pre-processing instructions for verifier generation"
        python src/create_verifiers_input.py \
            --instructions_file $AUGMENTED_INSTRUCTIONS_FILE \
            --output_file $VERIFIERS_INPUT_FILE
        if [ $? -ne 0 ]; then
            echo "Verifier pre-processing failed!"
            exit 1
        fi
        mark_step_completed $VER_PREPROCESSING
    else
        echo "VER: Pre-processing already completed, skipping."
    fi

    # Step 2.2: Submit verifiers inference job

    # Check if previous run failed and we need to resubmit
    if step_completed $VER_INFERENCE_FAILED && ! step_completed $VER_INFERENCE_COMPLETE; then
        echo "VER: Previous inference job failed, resubmitting..."
        submit_slurm_job "launch_generate_verifiers.sh" "VER_JOB_ID" "$VER_INFERENCE_SUBMITTED" "verifier generation"
        VER_JOB_ID=$LAST_SUBMITTED_JOB_ID
    elif ! step_completed $VER_INFERENCE_SUBMITTED; then
        submit_slurm_job "launch_generate_verifiers.sh" "VER_JOB_ID" "$VER_INFERENCE_SUBMITTED" "verifier generation"
        VER_JOB_ID=$LAST_SUBMITTED_JOB_ID
    else
        echo "VER: Inference job already submitted, retrieving job ID."
        VER_JOB_ID=$(python src/utils/checkpoint.py --checkpoint_file "$CHECKPOINT_FILE" get-job --prefix "VER_JOB_ID")
        echo "Retrieved Verifier Job ID: $VER_JOB_ID"
    fi

    # Step 2.3: Wait for verifiers inference job to complete
    if ! step_completed $VER_INFERENCE_COMPLETE; then
        echo "VER: Monitoring verifier generation job"
        echo "Current job ID: $VER_JOB_ID"
        VER_LOG_FILE="logs/${VER_JOB_ID}_verifiers.out"
        
        # First monitor the job status
        bash src/utils/monitor_slurm_job.sh \
            --job_id "$VER_JOB_ID" \
            --prefix "VER" \
            --check_interval 30
        
        if [ $? -ne 0 ]; then
            echo "VER: Job failed. Check logs and restart script."
            mark_step_completed $VER_INFERENCE_FAILED
            exit 1
        fi
        
        # Once job is completed, verify the output files
        python src/utils/verify_job_output.py \
            --input_file "$VERIFIERS_INPUT_FILE" \
            --output_file "$VERIFIERS_OUTPUT_FILE" \
            --log_file "$VER_LOG_FILE" \
            --required_completion 90 \
            --prefix "VER"
        
        if [ $? -ne 0 ]; then
            echo "VER: Job output verification failed. Check logs and restart script."
            mark_step_completed "$VER_INFERENCE_FAILED"
            exit 1
        fi
        
        # Mark step as completed in checkpoint file
        mark_step_completed "$VER_INFERENCE_COMPLETE"

        echo "VER: Inference job completed successfully!"
    else
        echo "VER: Inference job already completed, skipping."
    fi

    # Step 2.4: Cross-validate verifiers and filter
    if ! step_completed $VER_CROSS_VALIDATION; then
        echo "VER: Cross-validating and filtering verifiers"
        python src/verifiers_cross_validation.py \
            --verifiers_file $VERIFIERS_OUTPUT_FILE \
            --output_all_file $VERIFIERS_ALL_FILE \
            --output_filtered_file $VERIFIERS_FILTERED_FILE
        if [ $? -ne 0 ]; then
            echo "Cross-validation failed!"
            exit 1
        fi
        mark_step_completed $VER_CROSS_VALIDATION
    else
        echo "VER: Cross-validation already completed, skipping."
    fi

fi # End of Step 2 conditional execution

# Step 2.5: Concat queries
# Check if concat step should be skipped
if [[ "$SKIP_CONCAT" == "true" ]]; then
    echo "SKIP_CONCAT=true: Skipping Step 2.5 (query concatenation) and marking checkpoint as complete"
    
    # Mark concat checkpoint as completed
    mark_step_completed "$VER_QUERIES_CONCATED"
    
    echo "Step 2.5 skipped - using existing query concatenation file"
else
    echo "Starting Step 2.5: Concat queries"
fi

# Only execute Step 2.5 if not skipped
if [[ "$SKIP_CONCAT" != "true" ]]; then

if ! step_completed $VER_QUERIES_CONCATED; then
    echo "CONCAT: Concat queries"
    
    # Build concat_queries.py command with optional arguments
    CONCAT_CMD="python src/concat_queries.py \
        --verifiers_file $VERIFIERS_FILTERED_FILE \
        --output_file $VERIFIERS_QUERIES_FILE \
        --queries_dataset $QUERIES_DATASET \
        --query_column_name \"instruction\" \
        --response_column_name \"response\" \
        --query_max_len \"200\" \
        --queries_per_instruction 1"
    
    # Add optional arguments if they are set
    if [[ -n "$CONCAT_NUM_OUTPUT_LINES" ]]; then
        CONCAT_CMD="$CONCAT_CMD --num_of_output_lines $CONCAT_NUM_OUTPUT_LINES"
    fi
    if [[ -n "$CONCAT_MESSAGES_FORMAT" ]]; then
        CONCAT_CMD="$CONCAT_CMD --messages_format $CONCAT_MESSAGES_FORMAT"
    fi
    if [[ -n "$CONCAT_MESSAGES_KEY" ]]; then
        CONCAT_CMD="$CONCAT_CMD --messages_key $CONCAT_MESSAGES_KEY"
    fi
    if [[ -n "$CONCAT_TURNS" ]]; then
        CONCAT_CMD="$CONCAT_CMD --turns $CONCAT_TURNS"
    fi
    if [[ "$CONCAT_NO_FOLLOWUP" == "true" ]]; then
        CONCAT_CMD="$CONCAT_CMD --no-followup"
    fi
    
    # Execute the command
    eval $CONCAT_CMD
    if [ $? -ne 0 ]; then
        echo "Concat queries failed!"
        exit 1
    fi
    mark_step_completed $VER_QUERIES_CONCATED
else
    echo "CONCAT: Concat already performed, skipping."
fi

fi # End of Step 2.5 conditional execution

echo "Phase 1 pipeline completed successfully!"
echo "Generated verifiers are available at: $VERIFIERS_FILTERED_FILE"
echo "Query-instruction pairs with verifiers are available at: $VERIFIERS_QUERIES_FILE"

echo ""
echo "Starting Phase 2: Generate query responses and build SFT dataset"

# Phase 2 step definitions
RESP_INFERENCE_SUBMITTED="RESP_INFERENCE_SUBMITTED"
RESP_INFERENCE_COMPLETE="RESP_INFERENCE_COMPLETE"
RESP_INFERENCE_FAILED="RESP_INFERENCE_FAILED"
SFT_DATASET_BUILT="SFT_DATASET_BUILT"

# Step 3: Generate query responses
# Check if response generation step should be skipped
if [[ "$SKIP_RESPONSES" == "true" ]]; then
    echo "SKIP_RESPONSES=true: Skipping Step 3 (query response generation) and marking all Step 3 checkpoints as complete"
    
    # Mark all Step 3 checkpoints as completed
    mark_step_completed "$RESP_INFERENCE_SUBMITTED"
    mark_step_completed "$RESP_INFERENCE_COMPLETE"
    
    echo "Step 3 skipped - using existing response files"
else
    echo "Starting Step 3: Generate query responses"
fi

# Only execute Step 3 if not skipped
if [[ "$SKIP_RESPONSES" != "true" ]]; then

# Step 3.1: Submit query response generation job
if step_completed $RESP_INFERENCE_FAILED && ! step_completed $RESP_INFERENCE_COMPLETE; then
    echo "RESP: Previous inference job failed, resubmitting..."
    submit_slurm_job "launch_generate_query_responses.sh" "RESP_JOB_ID" "$RESP_INFERENCE_SUBMITTED" "query response generation"
    RESP_JOB_ID=$LAST_SUBMITTED_JOB_ID
elif ! step_completed $RESP_INFERENCE_SUBMITTED; then
    submit_slurm_job "launch_generate_query_responses.sh" "RESP_JOB_ID" "$RESP_INFERENCE_SUBMITTED" "query response generation"
    RESP_JOB_ID=$LAST_SUBMITTED_JOB_ID
else
    echo "RESP: Inference job already submitted, retrieving job ID."
    RESP_JOB_ID=$(python src/utils/checkpoint.py --checkpoint_file "$CHECKPOINT_FILE" get-job --prefix "RESP_JOB_ID")
    echo "Retrieved Response Generation Job ID: $RESP_JOB_ID"
fi

# Step 3.2: Wait for query response generation job to complete
if ! step_completed $RESP_INFERENCE_COMPLETE; then
    echo "RESP: Monitoring query response generation job"
    echo "Current job ID: $RESP_JOB_ID"
    RESP_LOG_FILE="logs/${RESP_JOB_ID}_responses.out"
    
    # First monitor the job status
    bash src/utils/monitor_slurm_job.sh \
        --job_id "$RESP_JOB_ID" \
        --prefix "RESP" \
        --check_interval 30
    
    if [ $? -ne 0 ]; then
        echo "RESP: Job failed. Check logs and restart script."
        mark_step_completed $RESP_INFERENCE_FAILED
        exit 1
    fi
    
    # Once job is completed, verify the output files
    python src/utils/verify_job_output.py \
        --input_file "$VERIFIERS_QUERIES_FILE" \
        --output_file "$SCORED_RESPONSES_FILE" \
        --log_file "$RESP_LOG_FILE" \
        --required_completion 90 \
        --prefix "RESP"
    
    if [ $? -ne 0 ]; then
        echo "RESP: Job output verification failed. Check logs and restart script."
        mark_step_completed "$RESP_INFERENCE_FAILED"
        exit 1
    fi
    
    # Mark step as completed in checkpoint file
    mark_step_completed "$RESP_INFERENCE_COMPLETE"

    echo "RESP: Inference job completed successfully!"
else
    echo "RESP: Inference job already completed, skipping."
fi

fi # End of Step 3 conditional execution

# Step 4: Build SFT dataset
echo "Starting Step 4: Build SFT dataset"

if ! step_completed $SFT_DATASET_BUILT; then
    echo "SFT: Building SFT dataset from scored responses"
    python src/build_sft.py \
        "$SCORED_RESPONSES_FILE" \
        --output "$SFT_DATASET_FILE" \
        --score_threshold "$SCORE_THRESHOLD"
    if [ $? -ne 0 ]; then
        echo "SFT dataset building failed!"
        exit 1
    fi
    mark_step_completed "$SFT_DATASET_BUILT"
    echo "SFT: Dataset built successfully!"
else
    echo "SFT: Dataset already built, skipping."
fi

echo ""
echo "Pipeline completed successfully!"
echo "Generated verifiers are available at: $VERIFIERS_FILTERED_FILE"
echo "Query-instruction pairs with verifiers are available at: $VERIFIERS_QUERIES_FILE"
echo "Scored responses are available at: $SCORED_RESPONSES_FILE"
echo "SFT dataset is available at: $SFT_DATASET_FILE"

deactivate