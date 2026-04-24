# LoRA Rank Sensitivity in Cross-Modal Fine-Tuning with ORCA

CS 639 Final Project - Testing whether ORCA's distributional alignment reduces intrinsic dimensionality, enabling LoRA to work at lower ranks.

## Hypothesis

ORCA's optimal transport-based alignment reduces the effective intrinsic dimensionality of cross-modal adaptation tasks, allowing LoRA to achieve competitive performance at lower ranks compared to baseline FPT.

## Setup

### Local (Mac)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd ORCA/src/otdd && pip install -e . && cd ../../..
```

### CHTC
```bash
bash setup.sh
```

## Experiments

### Local Testing (Synthetic Data)
```bash
python run_ecg_fpt_r2.py   # Baseline: No alignment
python run_ecg_orca_r2.py  # ORCA: With OTDD alignment
```

### CHTC (Real ECG Data)
```bash
# Single experiment
condor_submit submit_ecg_fpt_r2.sub

# All experiments (FPT + ORCA, ranks 2 & 4)
condor_submit submit_all.sub
```

## Architecture

- **Model**: RoBERTa-base (125M params)
- **Dataset**: PhysioNet Challenge 2017 ECG (4-class classification)
- **LoRA**: Applied to query & value attention matrices
- **Ranks Tested**: 2, 4
- **Trainable Params**: ~0.09% at rank=2 (73,728 / 85M)

## Key Files

- `ORCA/src/embedder.py` - LoRA integration (lines 135-214, 374-393)
- `ORCA/src/configs/ecg_*.yaml` - Experiment configs
- `run_ecg_*.py` - Experiment runners
- `submit_*.sub` - CHTC HTCondor files

## Results

Results are saved to `results/ecg_*_results.json` with full config and metrics.

## References

- ORCA: https://github.com/sjunhongshen/ORCA
- LoRA: https://arxiv.org/abs/2106.09685
- OTDD: https://arxiv.org/abs/2008.09758
