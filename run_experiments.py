#!/usr/bin/env python3
"""
Run all ORCA + LoRA ECG rank sensitivity experiments.

Conditions:
  - FPT (no alignment):  objective=l2, embedder_epochs=0
  - ORCA (OTDD alignment): objective=otdd-exact, embedder_epochs=5
  - LoRA ranks: 2, 4, 8, 16, 32
  - Seeds: 0, 1, 2

Usage:
  python run_experiments.py                  # Run ALL experiments
  python run_experiments.py --method fpt     # FPT only
  python run_experiments.py --method orca    # ORCA only
  python run_experiments.py --ranks 2 4      # Specific ranks
  python run_experiments.py --seeds 0 1      # Specific seeds
  python run_experiments.py --dry-run        # Print configs without running
"""

import sys
import os
import argparse
import time
import json
from datetime import datetime

# Package `otdd` lives under ORCA/src/otdd/otdd/ with setuptools root ORCA/src/otdd/.
# Putting only ORCA/src first makes Python load ORCA/src/otdd/ as otdd without the real subpackage.
_repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_repo_root, 'ORCA', 'src'))
sys.path.insert(0, os.path.join(_repo_root, 'ORCA', 'src', 'otdd'))

from munch import Munch
from main import main


def make_args(method, rank, seed):
    """Create experiment args for a given method/rank/seed combination."""

    if method == 'fpt':
        objective = 'l2'
        embedder_epochs = 0
        experiment_id = f'ecg_fpt_r{rank}'
    elif method == 'orca':
        objective = 'otdd-exact'
        embedder_epochs = 60
        experiment_id = f'ecg_orca_r{rank}'
    else:
        raise ValueError(f"Unknown method: {method}")

    return Munch(
        dataset='ECG',
        embedder_dataset='text',
        objective=objective,
        weight='roberta',
        maxsamples=128,

        experiment_id=experiment_id,
        seed=seed,
        epochs=15,
        embedder_epochs=embedder_epochs,
        predictor_epochs=0,
        finetune_method='all',
        drop_out=0,
        target_seq_len=64,

        # LoRA configuration
        use_lora=True,
        lora_rank=rank,
        lora_alpha=16,
        lora_dropout=0.1,

        batch_size=4,
        eval_batch_size=1000,
        accum=16,
        clip=-1,
        validation_freq=1,

        optimizer=Munch(
            name='SGD',
            params=Munch(lr=0.000001, betas=[0.9, 0.98], weight_decay=0.1, momentum=0.99)
        ),
        scheduler=Munch(
            name='WarmupLR',
            params=Munch(warmup_epochs=5, decay_epochs=200, sched=[20, 40, 60], base=0.2)
        ),
        no_warmup_scheduler=Munch(
            name='StepLR',
            params=Munch(warmup_epochs=10, decay_epochs=100, sched=[20, 40, 60], base=0.2)
        ),

        num_workers=4,
        reproducibility=False,
        valid_split=False,
    )


def run_single_experiment(method, rank, seed, dry_run=False):
    """Run a single experiment configuration."""
    label = f"{method.upper()} | rank={rank} | seed={seed}"
    print("\n" + "=" * 70)
    print(f"  EXPERIMENT: {label}")
    print("=" * 70)

    args = make_args(method, rank, seed)

    if dry_run:
        print(f"  objective:       {args.objective}")
        print(f"  embedder_epochs: {args.embedder_epochs}")
        print(f"  epochs:          {args.epochs}")
        print(f"  lora_rank:       {args.lora_rank}")
        print(f"  maxsamples:      {args.maxsamples}")
        print(f"  experiment_id:   {args.experiment_id}")
        print(f"  seed:            {args.seed}")
        print("  [DRY RUN - not executing]")
        return None

    start = time.time()
    try:
        result = main(False, args)
        elapsed = time.time() - start
        print(f"\n  Completed in {elapsed:.1f}s")
        return {
            'method': method,
            'rank': rank,
            'seed': seed,
            'result': result,
            'elapsed': elapsed,
            'status': 'success',
        }
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  FAILED after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return {
            'method': method,
            'rank': rank,
            'seed': seed,
            'result': None,
            'elapsed': elapsed,
            'status': 'failed',
            'error': str(e),
        }


def main_cli():
    parser = argparse.ArgumentParser(description='Run ORCA + LoRA ECG experiments')
    parser.add_argument('--method', choices=['fpt', 'orca', 'all'], default='all',
                        help='Which method to run (default: all)')
    parser.add_argument('--ranks', type=int, nargs='+', default=[2, 4, 8, 16, 32],
                        help='LoRA ranks to test (default: 2 4 8 16 32)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2],
                        help='Random seeds (default: 0 1 2)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print experiment configs without running')
    args = parser.parse_args()

    methods = ['fpt', 'orca'] if args.method == 'all' else [args.method]

    # Build experiment list
    experiments = []
    for method in methods:
        for rank in args.ranks:
            for seed in args.seeds:
                experiments.append((method, rank, seed))

    print("=" * 70)
    print(f"  ORCA + LoRA ECG Rank Sensitivity Experiments")
    print(f"  Methods: {methods}")
    print(f"  Ranks:   {args.ranks}")
    print(f"  Seeds:   {args.seeds}")
    print(f"  Total:   {len(experiments)} experiments")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    results = []
    for i, (method, rank, seed) in enumerate(experiments):
        print(f"\n>>> Experiment {i+1}/{len(experiments)}")
        result = run_single_experiment(method, rank, seed, dry_run=args.dry_run)
        if result:
            results.append(result)

    # Summary
    if not args.dry_run and results:
        print("\n" + "=" * 70)
        print("  EXPERIMENT SUMMARY")
        print("=" * 70)

        total_time = sum(r['elapsed'] for r in results)
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'failed']

        print(f"  Completed: {len(successful)}/{len(results)}")
        print(f"  Failed:    {len(failed)}")
        print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

        if failed:
            print("\n  Failed experiments:")
            for r in failed:
                print(f"    - {r['method'].upper()} rank={r['rank']} seed={r['seed']}: {r.get('error', 'unknown')}")

        # Save results summary
        summary_path = 'results/experiment_summary.json'
        os.makedirs('results', exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved to {summary_path}")

    print(f"\n  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main_cli()
