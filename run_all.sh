#!/bin/bash
# Master script to run all 4 experiments sequentially

echo "======================================================================"
echo "  Running All LoRA Experiments: FPT vs ORCA"
echo "======================================================================"
echo ""
echo "This will run 4 experiments:"
echo "  1. FPT + LoRA Rank=2 (no alignment)"
echo "  2. FPT + LoRA Rank=4 (no alignment)"
echo "  3. ORCA + LoRA Rank=2 (with alignment)"
echo "  4. ORCA + LoRA Rank=4 (with alignment)"
echo ""
echo "Estimated time: ~12-15 minutes on Mac MPS"
echo "======================================================================"
echo ""
read -p "Press Enter to start..."

# Activate venv
source venv/bin/activate

# Run experiments
echo ""
echo ">>> [1/4] Running FPT Rank=2..."
python run_fpt_r2.py
if [ $? -ne 0 ]; then
    echo "ERROR: FPT Rank=2 failed"
    exit 1
fi

echo ""
echo ">>> [2/4] Running FPT Rank=4..."
python run_fpt_r4.py
if [ $? -ne 0 ]; then
    echo "ERROR: FPT Rank=4 failed"
    exit 1
fi

echo ""
echo ">>> [3/4] Running ORCA Rank=2..."
python run_orca_r2.py
if [ $? -ne 0 ]; then
    echo "ERROR: ORCA Rank=2 failed"
    exit 1
fi

echo ""
echo ">>> [4/4] Running ORCA Rank=4..."
python run_orca_r4.py
if [ $? -ne 0 ]; then
    echo "ERROR: ORCA Rank=4 failed"
    exit 1
fi

# Plot results
echo ""
echo "======================================================================"
echo "  All experiments complete! Generating plots..."
echo "======================================================================"
python plot_results.py

echo ""
echo "======================================================================"
echo "  🎉 ALL DONE!"
echo "======================================================================"
echo ""
echo "Results saved in:"
echo "  - results/fpt_r2_results.json"
echo "  - results/fpt_r4_results.json"
echo "  - results/orca_r2_results.json"
echo "  - results/orca_r4_results.json"
echo "  - results/lora_comparison.png  ← THE GRAPH"
echo ""
echo "To view the graph:"
echo "  open results/lora_comparison.png"
echo ""
