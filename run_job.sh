#!/bin/bash
# CHTC job script for ORCA + LoRA ECG experiments
# Arguments: $1=method (fpt/orca), $2=rank, $3=seed

set -e

METHOD=$1
RANK=$2
SEED=$3

echo "=== Job Start: $(date) ==="
echo "Method: $METHOD, Rank: $RANK, Seed: $SEED"
echo "Host: $(hostname)"
nvidia-smi || echo "nvidia-smi not available"

# Unpack environment and code
tar -xzf venv.tar.gz
tar -xzf code.tar.gz

# Activate venv manually (avoid hardcoded VIRTUAL_ENV path)
export PATH="$(pwd)/venv/bin:$PATH"
export VIRTUAL_ENV="$(pwd)/venv"

# Verify python works
echo "Python: $(which python)"
python -c "import munch; print('munch OK')"

# Run experiment
python run_experiments.py --method $METHOD --ranks $RANK --seeds $SEED

# Pack results for transfer back
tar -czf results_${METHOD}_r${RANK}_s${SEED}.tar.gz results/

echo "=== Job End: $(date) ==="
