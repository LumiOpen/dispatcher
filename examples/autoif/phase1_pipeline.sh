#!/bin/bash

# Phase 1 Pipeline Script
# This script implements the pipeline described in phase1.md
LANGUAGE="en"

# Virtual environment settings
VENV_DIR=".venv"
REQUIREMENTS_FILE="requirements.txt"

# Data files
SEED_FILE="data/seed_instructions.txt"
AUGMENTED_INSTR_FILE="data/augmented_instructions.txt"
VERIFIERS_INPUT_FILE="data/verifiers_input.jsonl"
VERIFIERS_OUTPUT_FILE="data/verifiers_output.jsonl"
ALL_RESULTS_FILE="data/all_verifiers.jsonl"
FILTERED_VERIFIERS_FILE="data/filtered_verifiers.jsonl"
QUERIES_DATASET="databricks/databricks-dolly-15k"
VERIFIERS_QUERIES_FILE="data/verifiers_queries.jsonl"

# Temporary files
TMP_AUG_INPUT_FILE="data/tmp_aug_input.jsonl"
TMP_AUG_OUTPUT_FILE="data/tmp_aug_output.jsonl"

# Generation parameters 
NUM_PROMPTS=4

# Checkpointing mechanism
mkdir -p logs
CHECKPOINT_FILE="logs/phase1_checkpoint.log"
touch "$CHECKPOINT_FILE"

# Function to check if a step has been completed
step_completed() {
    local step_name="$1"
    run_python src/utils/checkpoint.py --checkpoint_file "$CHECKPOINT_FILE" check --step "$step_name"
    return $?
}

# Function to mark a step as completed
mark_step_completed() {
    local step_name="$1"
    run_python src/utils/checkpoint.py --checkpoint_file "$CHECKPOINT_FILE" mark --step "$step_name"
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
    run_python src/utils/checkpoint.py --checkpoint_file "$CHECKPOINT_FILE" save-job --prefix "$job_id_prefix" --job-id "$job_id"

    # set the job ID for further processing in the parent scope
    LAST_SUBMITTED_JOB_ID=$job_id
}

# Function to activate virtual environment and run a Python command
run_python() {
    # Run the python command in the virtual environment
    source "$VENV_DIR/bin/activate"
    python "$@"
    local result=$?
    deactivate
    return $result
}

# Function to set up virtual environment if it doesn't exist
setup_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        echo "Creating virtual environment at $VENV_DIR"
        
        # Check if Python is installed
        if ! command -v python3 &> /dev/null; then
            echo "ERROR: Python 3 is not installed!"
            exit 1
        fi
        
        # Create virtual environment
        python3 -m venv $VENV_DIR
        
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

echo "Starting Phase 1: Generating verifiers"

# Setup virtual environment
setup_venv

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
    run_python src/create_instructions_input.py \
        --seed_file $SEED_FILE \
        --output_file $TMP_AUG_INPUT_FILE \
        --num_prompts $NUM_PROMPTS
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
    submit_slurm_job "src/launch_scripts/augment_instructions_launch.sh" "AUG_JOB_ID" "$AUG_INFERENCE_SUBMITTED" "instruction augmentation"
    AUG_JOB_ID=$LAST_SUBMITTED_JOB_ID
elif ! step_completed $AUG_INFERENCE_SUBMITTED; then
    submit_slurm_job "src/launch_scripts/augment_instructions_launch.sh" "AUG_JOB_ID" "$AUG_INFERENCE_SUBMITTED" "instruction augmentation"
    AUG_JOB_ID=$LAST_SUBMITTED_JOB_ID
else
    echo "AUG: Inference job already submitted, retrieving job ID."
    AUG_JOB_ID=$(run_python src/utils/checkpoint.py --checkpoint_file "$CHECKPOINT_FILE" get-job --prefix "AUG_JOB_ID")
    echo "Retrieved Augmentation Job ID: $AUG_JOB_ID"
fi

# Step 1.3: Wait for inference job to complete
if ! step_completed $AUG_INFERENCE_COMPLETE; then
    echo "AUG: Waiting for inference job to complete"
    LOG_FILE="logs/${AUG_JOB_ID}_augment.out"
    
    # First monitor the job status
    run_python src/utils/monitor_slurm_job.py \
        --job_id "$AUG_JOB_ID" \
        --prefix "AUG" \
        --check_interval 30
    
    if [ $? -ne 0 ]; then
        echo "AUG: Job failed. Check logs and restart script."
        mark_step_completed $AUG_INFERENCE_FAILED
        exit 1
    fi
    
    # Once job is completed, verify the output files
    run_python src/utils/verify_job_output.py \
        --input_file "$TMP_AUG_INPUT_FILE" \
        --output_file "$TMP_AUG_OUTPUT_FILE" \
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
    run_python src/process_instructions_output.py \
        --input_file $TMP_AUG_OUTPUT_FILE \
        --output_file $AUGMENTED_INSTR_FILE \
        --seed_file $SEED_FILE \
        --language $LANGUAGE
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
VER_QUERIES_AUGMENTED="VER_QUERIES_AUGMENTED"

echo "Starting Step 2: Generate verifiers"

# Step 2.1: Create verifiers input
if ! step_completed $VER_PREPROCESSING; then
    echo "VER: Pre-processing instructions for verifier generation"
    run_python src/create_verifiers_input.py \
        --instructions_file $AUGMENTED_INSTR_FILE \
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
    submit_slurm_job "src/launch_scripts/generate_verifiers_launch.sh" "VER_JOB_ID" "$VER_INFERENCE_SUBMITTED" "verifier generation"
    VER_JOB_ID=$LAST_SUBMITTED_JOB_ID
elif ! step_completed $VER_INFERENCE_SUBMITTED; then
    submit_slurm_job "src/launch_scripts/generate_verifiers_launch.sh" "VER_JOB_ID" "$VER_INFERENCE_SUBMITTED" "verifier generation"
    VER_JOB_ID=$LAST_SUBMITTED_JOB_ID
else
    echo "VER: Inference job already submitted, retrieving job ID."
    VER_JOB_ID=$(run_python src/utils/checkpoint.py --checkpoint_file "$CHECKPOINT_FILE" get-job --prefix "VER_JOB_ID")
    echo "Retrieved Verifier Job ID: $VER_JOB_ID"
fi

# Step 2.3: Wait for verifiers inference job to complete
if ! step_completed $VER_INFERENCE_COMPLETE; then
    echo "VER: Monitoring verifier generation job"
    echo "Current job ID: $VER_JOB_ID"
    VER_LOG_FILE="logs/${VER_JOB_ID}_verifiers.out"
    
    # First monitor the job status
    run_python src/utils/monitor_slurm_job.py \
        --job_id "$VER_JOB_ID" \
        --prefix "VER" \
        --check_interval 30
    
    if [ $? -ne 0 ]; then
        echo "VER: Job failed. Check logs and restart script."
        mark_step_completed $VER_INFERENCE_FAILED
        exit 1
    fi
    
    # Once job is completed, verify the output files
    run_python src/utils/verify_job_output.py \
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
    run_python src/verifiers_cross_validation.py \
        --verifiers_file $VERIFIERS_OUTPUT_FILE \
        --all_results_file $ALL_RESULTS_FILE \
        --filtered_file $FILTERED_VERIFIERS_FILE \
        --output_file $VERIFIERS_QUERIES_FILE \
        --queries_dataset $QUERIES_DATASET \
        --queries_per_instruction 16 \
        --verifiers_per_query 10
    if [ $? -ne 0 ]; then
        echo "Cross-validation failed!"
        exit 1
    fi
    mark_step_completed $VER_CROSS_VALIDATION
    mark_step_completed $VER_QUERIES_AUGMENTED  # Since the script handles both steps
else
    echo "VER: Cross-validation already completed, skipping."
fi

echo "Phase 1 pipeline completed successfully!"
echo "Generated verifiers are available at: $FILTERED_VERIFIERS_FILE"
echo "Query-instruction pairs with verifiers are available at: $VERIFIERS_QUERIES_FILE"