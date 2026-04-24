# 🚀 READY TO RUN

Everything is set up! Here's what you have:

## ✅ What's Done

1. **LoRA Integration** - Added to ORCA codebase (Swin & RoBERTa)
2. **4 Config Files** - FPT vs ORCA, Rank 2 vs 4
3. **4 Run Scripts** - Individual experiments
4. **1 Master Script** - Runs all 4 sequentially
5. **1 Plotting Script** - Creates comparison graph
6. **MPS Support** - Runs on your Mac GPU
7. **All Dependencies** - Installed in venv/

## 🎯 Quick Start (Choose One)

### Option A: Run All 4 Experiments (~12-15 min)
```bash
cd /Users/adisrinivasan/Desktop/639/final_project
./run_all.sh
```

### Option B: Run Individual Experiments
```bash
cd /Users/adisrinivasan/Desktop/639/final_project

# Run one at a time
python run_fpt_r2.py      # ~2 min
python run_fpt_r4.py      # ~2 min
python run_orca_r2.py     # ~4 min (alignment takes time)
python run_orca_r4.py     # ~4 min

# Then plot
python plot_results.py
open results/lora_comparison.png
```

## 📊 Expected Output

### During Each Run:
```
============================================================
Adding LoRA to Swin Transformer
============================================================
LoRA Configuration:
  Rank: 2
  Alpha: 16
  Dropout: 0.1
  Target modules: ['query', 'value']
  Trainable params: 295,424 / 87,768,100 (0.34%)
============================================================

target_seq_len 512
src feat shape torch.Size([50000, 1024]) torch.Size([50000]) num classes 100

------- Experiment Summary --------
id: fpt_r2
dataset: CIFAR100    batch size: 8    lr: 0.0001
num train batch: 16    num validation batch: 25    num test batch: 100
finetune method: all
param count: 88,xxxxx trainable params: 1,xxxxx

------- Start Training --------
[train full 0 0.000100 ] time elapsed: XX.Xs    train loss: X.XXX    val loss: X.XXX    val score: 0.XXX

------- Start Test --------
[test best-validated]    time elapsed: X.Xs    test loss: X.XXX    test score: 0.XXX

✅ FPT RANK=2 COMPLETE
  Test Accuracy: XX.XX%
  Time: XX.Xs
  Results saved to: results/fpt_r2_results.json
```

### After All 4 Complete:
```
======================================================================
  RESULTS SUMMARY
======================================================================
Condition            Rank       Accuracy        Time (s)
----------------------------------------------------------------------
FPT (NoAlign)        2           XX.XX%            XX.X
FPT (NoAlign)        4           XX.XX%            XX.X
ORCA (Align)         2           XX.XX%            XX.X
ORCA (Align)         4           XX.XX%            XX.X
======================================================================

📊 ORCA vs FPT Improvement:
  Rank=2: +X.XX% (ORCA better)
  Rank=4: +X.XX% (ORCA better)

✅ Analysis complete!
   View graph: results/lora_comparison.png
```

## 📈 The Graph

`results/lora_comparison.png` shows:
- **Left**: Line plot of accuracy vs rank
- **Right**: Bar chart comparing FPT vs ORCA
- **Red line/bars**: FPT (no alignment)
- **Blue line/bars**: ORCA (with alignment)

## 🔬 What You're Testing

**Hypothesis**: ORCA's alignment (Stage 2) reduces intrinsic dimensionality, so LoRA achieves better performance at lower ranks.

**Conditions**:
1. **FPT + LoRA**: No alignment (baseline)
   - `objective: l2` in configs
2. **ORCA + LoRA**: With OTDD alignment
   - `objective: otdd-exact` in configs

**Ranks**: 2 and 4 (just validation today)

## 📁 Output Files

After running:
```
results/
├── fpt_r2_results.json
├── fpt_r4_results.json
├── orca_r2_results.json
├── orca_r4_results.json
├── lora_comparison.png  ← THE GRAPH
└── CIFAR100/            ← Model checkpoints
    ├── all_fpt_r2/
    ├── all_fpt_r4/
    ├── all_orca_r2/
    └── all_orca_r4/
```

## ⚠️ If Something Fails

1. **"ModuleNotFoundError"**: Activate venv first
   ```bash
   source venv/bin/activate
   ```

2. **"CUDA not available"**: Expected on Mac - uses MPS instead

3. **"Slow downloads"**: HuggingFace models (Swin-base ~350MB) download once

4. **"OOM error"**: Reduce batch_size in config (currently 8)

## ✅ You'll Know It Worked If:

1. ✓ All 4 scripts complete without errors
2. ✓ You see LoRA param counts printed (e.g., "Trainable: 295,424 / 87,768,100 (0.34%)")
3. ✓ Training loss decreases during epochs
4. ✓ Test accuracy is reported (not 0%)
5. ✓ 4 JSON files created in results/
6. ✓ PNG graph created and looks sensible

## 🚀 Next Steps (After Validation)

1. Scale up ranks: {2, 4, 8, 16, 32}
2. Increase epochs and use full dataset
3. Test on ECG (1D time series)
4. Push production runs to CHTC

---

**Ready to go! Just run:**
```bash
cd /Users/adisrinivasan/Desktop/639/final_project
./run_all.sh
```

or start with one experiment:
```bash
python run_fpt_r2.py
```
