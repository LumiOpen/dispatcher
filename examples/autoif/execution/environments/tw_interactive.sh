# TensorWave Interactive Environment Setup
# This is sourced by job templates to set up the container environment

# Source the singularity launcher (sets up container, binds, etc.)
source /shared_silo/scratch/adamhrin@amd.com/dispatcher/examples/autoif/execution/environments/singularity_launcher.sh

# Install dispatcher package into container's user site-packages
echo "Installing dispatcher package..."
run_sing_python -m pip install --user --upgrade \
    -e /shared_silo/scratch/adamhrin@amd.com/dispatcher \
    fasttext-numpy2 2>/dev/null || true
echo "Dispatcher installation complete."
