# LUMI Environment Setup using Module System
# Users can customize this for their specific environment

# Clean environment
unset VIRTUAL_ENV
unset PYTHONHOME
unset PYTHONPATH
unset PYTHONSTARTUP
unset PYTHONNOUSERSITE
unset PYTHONEXECUTABLE

# Set up environment
mkdir -p logs pythonuserbase
export PYTHONUSERBASE="$(pwd)/pythonuserbase"

# Activate virtual environment for task dependencies
VENV_DIR="{{ venv_dir }}"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "ERROR: Virtual environment not found at $VENV_DIR"
    exit 1
fi

pip install -r requirements-vultr.txt
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

export HF_HOME="{{ hf_home }}"
