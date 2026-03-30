###
# configure the following.
export LANGUAGE="${language:-fi}"
export MODEL=${model:-DeepSeek-V3}
INPUT_FILE=${input_file:-/scratch/project_462000353/adamhrin/dispatcher/examples/translation/data/default-train-sample-100_translations_DeepSeek-V3_fi.jsonl}

MODEL_NAME=$(basename "$MODEL")
DATADIR=$(dirname "$INPUT_FILE")
FILE_NAME=$(basename "$INPUT_FILE" .jsonl)
OUTPUT_FILE=${DATADIR}/${FILE_NAME}_traces_${MODEL_NAME}_${LANGUAGE}.jsonl
TASK=tasks.traces_task.TracesTask

echo "Using model: $MODEL"
echo "Model base name: $MODEL_NAME"
echo "Language: $LANGUAGE"
echo "Input file: $INPUT_FILE"
echo "Output file: $OUTPUT_FILE"

# generation parameters
# These should be tuned so that you do not overload your backend vllm server,
# or run into any timeouts.  timeouts greatly affect the efficiency of the
# workflow.
WORKERS=${workers:-16}          # number of simultaneous backend requests
BATCH_SIZE=${batch_size:-1}        # amount of work to request from dispatcher. 1 is usually fine.

# Timeouts are safety valves and you should not hit them in the normal course
# of your workflow.  if you do, it suggests you need to change something about
# your configuration--tasks are usually written to expect success.
REQUEST_TIMEOUT=${request_timeout:-3600} # adjust as needed for your task so that you do not hit
WORK_TIMEOUT=${work_timeout:-7200}   # time for dispatcher to give up on a work item and reissue it.  ideally this should never be hit.

#
# If you are changing the model, be sure to update GPUS_PER_TASK and the
# sbatch --ntasks-per-node configuration appropriately.
# Typically on Lumi 70B = 4 GPUs, 34B = 2 GPUs, 8B = 1 GPU
# --ntasks-per-node should be int(8 / GPUS_PER_TASK)
#

GPUS_PER_TASK=${gpus_per_task:-8}     # enough for the model and large batch size
MAX_MODEL_LEN=${max_model_len:-16384} # for efficiency, only as much as you think you need for efficiency

VLLM_HOST="${vllm_host:-127.0.0.1}"
VLLM_PORT="${vllm_port:-8000}"

# end configuration
###################

# clean up any venv that might be inherited from the launch environment.
unset VIRTUAL_ENV
unset PYTHONHOME
unset PYTHONPATH
unset PYTHONSTARTUP
unset PYTHONNOUSERSITE
unset PYTHONEXECUTABLE

pip3 install --user git+https://github.com/LumiOpen/dispatcher.git

#cd /scratch/project_462000353/zosaelai2/LumiOpen/translation/dispatcher
#pip install -e .
# pip install fasttext
#cd /scratch/project_462000353/zosaelai2/LumiOpen/translation/dispatcher/examples/translation

# dispatcher server will run on the first node, before we launch the worker
# tasks.
export DISPATCHER_SERVER=$(hostname)
export DISPATCHER_PORT=9999
python3 -m dispatcher.server \
    --infile $INPUT_FILE \
    --outfile $OUTPUT_FILE \
    --work-timeout $WORK_TIMEOUT \
    --host 0.0.0.0 \
    --port ${DISPATCHER_PORT} &

sleep 10

python3 -m dispatcher.taskmanager.cli \
    --dispatcher ${DISPATCHER_SERVER}:${DISPATCHER_PORT} \
    --task "$TASK" \
    --batch-size "$BATCH_SIZE" \
    --workers "$WORKERS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --host "$VLLM_HOST" \
    --port "$VLLM_PORT" \
    --model "$MODEL" \
    --request-timeout "$REQUEST_TIMEOUT" \
    --no-launch
    # --silence-vllm-logs


