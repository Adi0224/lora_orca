# LoRA Rank Sensitivity in Cross-Modal Fine-Tuning with ORCA

CS 639 Final Project - Testing whether ORCA's distributional alignment reduces intrinsic dimensionality, enabling LoRA to work at lower ranks.

## Hypothesis

ORCA's optimal transport-based alignment reduces the effective intrinsic dimensionality of cross-modal adaptation tasks, allowing LoRA to achieve competitive performance at lower ranks compared to baseline FPT.

## Setup

```bash
git clone https://github.com/Adi0224/lora_orca.git
cd lora_orca
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd ORCA/src/otdd && pip install -e . && cd ../../..
```

The ORCA source code is included directly in this repo (modified from [Shen et al., 2023](https://arxiv.org/abs/2302.05738) with LoRA integration and bug fixes).

The ECG dataset (`challenge2017.pkl`) and text alignment data (`text_xs.npy`, `text_ys.npy`) should be placed in `datasets/`.

## Running Experiments

All experiments are run through `run_experiments.py`:

```bash
# Dry run (print configs without executing)
python run_experiments.py --dry-run

# Run everything (30 experiments: 2 methods x 5 ranks x 3 seeds)
python run_experiments.py

# Run a single method
python run_experiments.py --method fpt
python run_experiments.py --method orca

# Run specific ranks and seeds
python run_experiments.py --method fpt --ranks 2 4 8 16 32 --seeds 0 1
python run_experiments.py --method orca --ranks 16 32 --seeds 0 1 2
```

### Experiment Conditions

| Condition | Objective | Embedder Epochs | Description |
|-----------|-----------|-----------------|-------------|
| FPT | l2 | 0 | No distributional alignment (baseline) |
| ORCA | otdd-exact | 5 | OTDD alignment before fine-tuning |

### LoRA Ranks

2, 4, 8, 16, 32 -- each run with seeds 0, 1, 2.

## Architecture

- **Model**: RoBERTa-base (125M params)
- **Dataset**: PhysioNet Challenge 2017 ECG (4-class: Normal, AFib, Other, Noisy)
- **LoRA**: Applied to query and value attention matrices via PEFT
- **Trainable Params**: ~0.6% (512K / 85M) including LoRA + embedder + predictor

### ORCA Pipeline

1. **Stage 1**: Train input embedder to map ECG signals into RoBERTa's token space
2. **Stage 2** (ORCA only): Align ECG and text embedding distributions via OTDD
3. **Stage 3**: Fine-tune with only LoRA adapters + embedder + predictor trainable (base RoBERTa frozen)

## Key Files

- `run_experiments.py` - Unified experiment runner for all conditions
- `test_lora_freeze.py` - Verifies LoRA parameter freezing works correctly
- `plot_results.py` - Generates comparison plots from results
- `ORCA/src/embedder.py` - LoRA integration via PEFT
- `ORCA/src/task_configs.py` - Optimizer/scheduler setup with LoRA-aware freezing
- `ORCA/src/main.py` - Main training loop
- `ORCA/src/configs/ecg_*.yaml` - Per-condition config files

## Results

Results are saved to `results/ECG/all_<experiment_id>/<seed>/` with model checkpoints, training curves, and test scores. A summary JSON is written to `results/experiment_summary.json` after each batch.

## References

- LoRA: https://arxiv.org/abs/2106.09685
- OTDD: https://arxiv.org/abs/2008.09758

```bibtex
@inproceedings{shen2023orca,
  author = {Shen, Junhong and Li, Liam and Dery, Lucio M. and Staten, Corey and Khodak, Mikhail and Neubig, Graham and Talwalkar, Ameet},
  title = {Cross-Modal Fine-Tuning: Align then Refine},
  publisher = {ICML},
  year = {2023},
  url = {https://arxiv.org/abs/2302.05738}
}
```
