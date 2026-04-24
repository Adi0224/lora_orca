#!/usr/bin/env python3
"""
ECG: ORCA + LoRA Rank 4 (With Alignment)
"""
import sys
import os
import json
import time

def dict_to_munch(d):
    from munch import Munch
    if isinstance(d, dict):
        return Munch({k: dict_to_munch(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_munch(item) for item in d]
    else:
        return d

if __name__ == '__main__':
    sys.path.insert(0, 'ORCA/src')

    print("="*70)
    print("  ECG: ORCA + LoRA RANK=4 (RoBERTa, With Alignment)")
    print("="*70)
    print()

    import yaml
    from munch import Munch
    from main import main

    with open('ORCA/src/configs/ecg_orca_r4.yaml', 'r') as f:
        config = yaml.safe_load(f)
        args = dict_to_munch(config['hyperparameters'])

    os.makedirs('results', exist_ok=True)

    start_time = time.time()
    print(f"Starting experiment: {args.experiment_id}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Model: RoBERTa-base")
    print(f"  Objective: {args.objective} (WITH OTDD alignment)")
    print(f"  LoRA Rank: {args.lora_rank}")
    print(f"  Epochs: {args.epochs}")
    print()

    try:
        main(False, args)
        elapsed = time.time() - start_time

        test_score_path = f'results/{args.dataset}/{args.finetune_method}_{args.experiment_id}/{args.seed}/test_score.npy'
        if os.path.exists(test_score_path):
            import numpy as np
            test_scores = np.load(test_score_path)
            test_acc = test_scores[-1]
        else:
            test_acc = 0.0

        results = {
            'experiment_id': args.experiment_id,
            'dataset': 'ECG',
            'model': 'RoBERTa',
            'condition': 'ORCA',
            'rank': args.lora_rank,
            'alignment': True,
            'test_accuracy': float(test_acc),
            'time_seconds': elapsed,
            'config': dict(args)
        }

        with open('results/ecg_orca_r4_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        print()
        print("="*70)
        print(f"✅ ECG ORCA RANK=4 COMPLETE")
        print(f"  Test Accuracy: {test_acc*100:.2f}%")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Results saved to: results/ecg_orca_r4_results.json")
        print("="*70)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
