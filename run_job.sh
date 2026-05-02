#!/bin/bash
# CHTC: $1=method $2=rank $3=seed  → run_experiments.py
#
# Completed (method,rank,seed) combos are skipped in run_experiments.py via SKIP_COMPLETED_COMBOS.
# Re-run listed combos anyway: FORCE_RERUN_LISTED=1
#
# Tesla P100 (Pascal/sm_60): build venv with requirements-chtc.txt torch+cu118 wheels.

set -e

METHOD=$1
RANK=$2
SEED=$3

echo "=== Job Start: $(date) ==="
echo "Method: $METHOD, Rank: $RANK, Seed: $SEED"
echo "Host: $(hostname)"
nvidia-smi || echo "nvidia-smi not available"

tar -xzf venv.tar.gz
tar -xzf code.tar.gz

export PATH="$(pwd)/venv/bin:$PATH"
export VIRTUAL_ENV="$(pwd)/venv"

mkdir -p results

echo "Python: $(which python)"
python -c "import munch; print('munch OK')"

# Fail fast if this PyTorch wheel cannot execute CUDA kernels on this GPU.
python <<'PY'
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    x = torch.randn(4, 8, 64, device="cuda")
    y = torch.nn.Conv1d(8, 4, kernel_size=3, padding=1).cuda()(x)
    print("cuda smoke OK:", y.shape, "|", torch.cuda.get_device_name(0))
PY

RERUN=()
if [ "${FORCE_RERUN_LISTED:-0}" = 1 ]; then
  RERUN=(--rerun-listed)
fi

python run_experiments.py \
  "${RERUN[@]}" \
  --method "$METHOD" \
  --ranks "$RANK" \
  --seeds "$SEED"

tar -czf "results_${METHOD}_r${RANK}_s${SEED}.tar.gz" results/

echo "=== Job End: $(date) ==="
