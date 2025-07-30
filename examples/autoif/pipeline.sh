#!/bin/bash

# AutoIF Pipeline Script
RUNID="ifeval"

# Default values
LANGUAGE="en"
MODEL="meta-llama/Llama-3.3-70B-Instruct"  # Default model
SEED_FILE_DEFAULT="data/seed_instructions_${RUNID}.txt"
AUGMENTED_INSTRUCTIONS_FILE_DEFAULT="data/augmented_instructions_${RUNID}.csv"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            echo "Usage: $0 [OPTIONS] [model_path]"
            echo "AutoIF Pipeline Script - Runs both Phase 1 (verifier generation) and Phase 2 (response generation & SFT dataset creation)"
            echo "Options:"
            echo "  --seed <path>           Path to seed instructions file"
            echo "  --augmented <path>      Path to augmented instructions file"
            echo "  --help                  Show this help message"
            echo ""
            echo "Arguments:"
            echo "  model_path              Specify the model to use (default: 'meta-llama/Llama-3.3-70B-Instruct')"
            exit 0
            ;;
        --seed)
            SEED_FILE_CUSTOM="$2"
            shift 2
            ;;
        --augmented)
            AUGMENTED_INSTRUCTIONS_FILE_CUSTOM="$2"
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

# Virtual environment settings
VENV_DIR=".venv"
REQUIREMENTS_FILE="requirements.txt"

# Data files - use custom paths if provided, otherwise use defaults
SEED_FILE="${SEED_FILE_CUSTOM:-$SEED_FILE_DEFAULT}"
AUGMENTED_INSTRUCTIONS_FILE="${AUGMENTED_INSTRUCTIONS_FILE_CUSTOM:-$AUGMENTED_INSTRUCTIONS_FILE_DEFAULT}"
echo "Using seed instructions file: $SEED_FILE"
echo "Using augmented instructions file: $AUGMENTED_INSTRUCTIONS_FILE"
VERIFIERS_ALL_FILE="data/verifiers_all_${RUNID}.jsonl"
VERIFIERS_FILTERED_FILE="data/verifiers_filtered_${RUNID}.jsonl"
VERIFIERS_QUERIES_FILE="data/verifiers_queries_${RUNID}.jsonl"
SCORED_RESPONSES_FILE="data/scored_responses_${RUNID}.jsonl"
SFT_DATASET_FILE="data/sft_dataset_${RUNID}.jsonl"
QUERIES_DATASET="databricks/databricks-dolly-15k"
# export to make available for launch script
export VERIFIERS_INPUT_FILE="data/verifiers_input_${RUNID}.jsonl" 
export VERIFIERS_OUTPUT_FILE="data/verifiers_output_${RUNID}.jsonl"
export AUGMENT_INPUT_FILE="data/aug_input_${RUNID}.jsonl"
export AUGMENT_OUTPUT_FILE="data/aug_output_${RUNID}.jsonl"
export VERIFIERS_QUERIES_FILE
export SCORED_RESPONSES_FILE

# Export MODEL to make it available for launch scripts
export MODEL
# make LANGUAGE available for launch scripts
export LANGUAGE
export HF_HOME="/scratch/project_462000353/hf_cache"
export FUNCTION_TIMEOUT
export RESP_WORKERS
export RESP_REQUEST_TIMEOUT
export RESP_WORK_TIMEOUT

# Other config
NUM_OF_AUGMENTED_INSTRUCTIONS=2

# Verifier generation config
export FUNCTION_TIMEOUT=5  # seconds
export MIN_FUNCTIONS=1  # original autoif: 3
export MIN_TEST_CASES=1  # original autoif: 10
export ACCURACY_THRESHOLD=0.8

# Phase 2 config
SCORE_THRESHOLD=3  # minimum score for responses to be included in SFT dataset
RESP_WORKERS=32    # number of simultaneous backend requests for response generation
RESP_REQUEST_TIMEOUT=600  # timeout for individual requests
RESP_WORK_TIMEOUT=1800    # timeout for work items

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

echo "Starting Step 1: Augment instructions"

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

# Step 2: Generate verifiers
VER_PREPROCESSING="VER_PREPROCESSING"
VER_INFERENCE_SUBMITTED="VER_INFERENCE_SUBMITTED"
VER_INFERENCE_COMPLETE="VER_INFERENCE_COMPLETE"
VER_INFERENCE_FAILED="VER_INFERENCE_FAILED"
VER_CROSS_VALIDATION="VER_CROSS_VALIDATION"
VER_QUERIES_CONCATED="VER_QUERIES_CONCATED"

echo "Starting Step 2: Generate verifiers"

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

# Step 2.5: Concat queries
if ! step_completed $VER_QUERIES_CONCATED; then
    echo "VER: Concat queries"
    python src/concat_queries.py \
        --verifiers_file $VERIFIERS_FILTERED_FILE \
        --output_file $VERIFIERS_QUERIES_FILE \
        --queries_dataset $QUERIES_DATASET \
        --query_column_name "instruction" \
        --response_column_name "response" \
        --query_max_len "200" \
        --queries_per_instruction 1
    if [ $? -ne 0 ]; then
        echo "Concat queries failed!"
        exit 1
    fi
    mark_step_completed $VER_QUERIES_CONCATED
else
    echo "VER: Concat already performed, skipping."
fi

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
echo "Starting Step 3: Generate query responses"

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