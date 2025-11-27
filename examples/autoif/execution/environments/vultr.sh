# Vultr Environment Setup
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

# Activate virtual environment for task dependencies (optional)
VENV_DIR="{{ venv_dir | default('') }}"
if [ -n "$VENV_DIR" ] && [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
    echo "Activated virtual environment: $VENV_DIR"
elif [ -n "$VENV_DIR" ]; then
    echo "Warning: Virtual environment not found at $VENV_DIR (continuing without venv)"
else
    echo "Info: No virtual environment configured (venv_dir not set)"
fi

# Install requirements if file is set and exists
REQUIREMENTS_FILE="{{ requirements_file | default('') }}"
if [ -n "$REQUIREMENTS_FILE" ] && [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Installing requirements from $REQUIREMENTS_FILE..."
    pip install --user -r "$REQUIREMENTS_FILE" || echo "Warning: Some requirements failed to install (continuing anyway)"
elif [ -n "$REQUIREMENTS_FILE" ]; then
    echo "Warning: Requirements file not found at $REQUIREMENTS_FILE (skipping pip install)"
else
    echo "Info: No requirements file configured (requirements_file not set)"
fi

# Add current directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

# Set HF_HOME if configured
HF_HOME_VALUE="{{ hf_home | default('') }}"
if [ -n "$HF_HOME_VALUE" ]; then
    export HF_HOME="$HF_HOME_VALUE"
    echo "HF_HOME set to: $HF_HOME"
else
    echo "Info: HF_HOME not configured"
fi
