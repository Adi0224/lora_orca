#!/usr/bin/env python3
"""
Plot comparison of FPT vs ORCA with LoRA at different ranks
"""
import json
import os
import matplotlib.pyplot as plt
import numpy as np

print("="*70)
print("  Plotting Results: FPT vs ORCA + LoRA")
print("="*70)
print()

# Load all results
results_files = {
    'FPT Rank=2': 'results/fpt_r2_results.json',
    'FPT Rank=4': 'results/fpt_r4_results.json',
    'ORCA Rank=2': 'results/orca_r2_results.json',
    'ORCA Rank=4': 'results/orca_r4_results.json',
}

results = {}
for name, filepath in results_files.items():
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            results[name] = json.load(f)
        print(f"✓ Loaded: {name}")
    else:
        print(f"✗ Missing: {name} (file: {filepath})")

if len(results) == 0:
    print("\n❌ No results found! Run the experiments first.")
    exit(1)

print()

# Extract data
fpt_ranks = []
fpt_accs = []
orca_ranks = []
orca_accs = []

for name, data in results.items():
    rank = data['rank']
    acc = data['test_accuracy'] * 100  # Convert to percentage

    if data['condition'] == 'FPT':
        fpt_ranks.append(rank)
        fpt_accs.append(acc)
    else:  # ORCA
        orca_ranks.append(rank)
        orca_accs.append(acc)

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('LoRA Rank Sensitivity: FPT vs ORCA on CIFAR-100', fontsize=14, fontweight='bold')

# Plot 1: Test Accuracy vs Rank
ax1.plot(fpt_ranks, fpt_accs, 'o-', color='#e74c3c', linewidth=2, markersize=10, label='FPT (No Alignment)')
ax1.plot(orca_ranks, orca_accs, 's-', color='#3498db', linewidth=2, markersize=10, label='ORCA (With Alignment)')
ax1.set_xlabel('LoRA Rank', fontsize=12)
ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
ax1.set_title('Performance vs Rank', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xticks([2, 4])

# Add value labels on points
for rank, acc in zip(fpt_ranks, fpt_accs):
    ax1.annotate(f'{acc:.1f}%', (rank, acc), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
for rank, acc in zip(orca_ranks, orca_accs):
    ax1.annotate(f'{acc:.1f}%', (rank, acc), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=9)

# Plot 2: Bar chart comparison
x = np.arange(len([2, 4]))
width = 0.35

fpt_vals = [fpt_accs[fpt_ranks.index(r)] if r in fpt_ranks else 0 for r in [2, 4]]
orca_vals = [orca_accs[orca_ranks.index(r)] if r in orca_ranks else 0 for r in [2, 4]]

bars1 = ax2.bar(x - width/2, fpt_vals, width, label='FPT', color='#e74c3c', alpha=0.8)
bars2 = ax2.bar(x + width/2, orca_vals, width, label='ORCA', color='#3498db', alpha=0.8)

ax2.set_xlabel('LoRA Rank', fontsize=12)
ax2.set_ylabel('Test Accuracy (%)', fontsize=12)
ax2.set_title('Side-by-Side Comparison', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(['Rank=2', 'Rank=4'])
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('results/lora_comparison.png', dpi=300, bbox_inches='tight')
print(f"✅ Graph saved to: results/lora_comparison.png")

# Print summary table
print("\n" + "="*70)
print("  RESULTS SUMMARY")
print("="*70)
print(f"{'Condition':<20} {'Rank':<10} {'Accuracy':<15} {'Time (s)':<10}")
print("-"*70)

for name in ['FPT Rank=2', 'FPT Rank=4', 'ORCA Rank=2', 'ORCA Rank=4']:
    if name in results:
        data = results[name]
        cond = f"{data['condition']} ({'Align' if data['alignment'] else 'NoAlign'})"
        print(f"{cond:<20} {data['rank']:<10} {data['test_accuracy']*100:>6.2f}%{'':<8} {data['time_seconds']:>8.1f}")

print("="*70)

# Calculate improvements
if len(fpt_ranks) == 2 and len(orca_ranks) == 2:
    print("\n📊 ORCA vs FPT Improvement:")
    for rank in [2, 4]:
        if rank in fpt_ranks and rank in orca_ranks:
            fpt_acc = fpt_accs[fpt_ranks.index(rank)]
            orca_acc = orca_accs[orca_ranks.index(rank)]
            improvement = orca_acc - fpt_acc
            print(f"  Rank={rank}: {improvement:+.2f}% {'(ORCA better)' if improvement > 0 else '(FPT better)'}")

print("\n✅ Analysis complete!")
print(f"   View graph: results/lora_comparison.png")
