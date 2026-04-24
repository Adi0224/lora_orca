#!/bin/bash
# CHTC Job Execution Script

set -e  # Exit on error

EXPERIMENT=$1

echo "=========================================="
echo "Running experiment: $EXPERIMENT"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "=========================================="

# Run setup
bash setup.sh

# Activate virtual environment
source venv/bin/activate

# Run the experiment
echo "Starting Python script..."
python run_${EXPERIMENT}.py

# Package results
echo "Packaging results..."
mkdir -p output
cp -r results/* output/ 2>/dev/null || true
cp results/${EXPERIMENT}_results.json output/ 2>/dev/null || true

# Create a summary
echo "=========================================="
echo "Experiment Complete: $EXPERIMENT"
echo "Date: $(date)"
if [ -f "results/${EXPERIMENT}_results.json" ]; then
    echo "Results:"
    cat "results/${EXPERIMENT}_results.json"
fi
echo "=========================================="

exit 0
