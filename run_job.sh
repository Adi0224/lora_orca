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
source venv/bin/activate

# Install otdd in-place
cd ORCA/src/otdd && pip install -e . --quiet && cd ../../..

# Run experiment
python run_experiments.py --method $METHOD --ranks $RANK --seeds $SEED

# Pack results for transfer back
tar -czf results_${METHOD}_r${RANK}_s${SEED}.tar.gz results/

echo "=== Job End: $(date) ==="
