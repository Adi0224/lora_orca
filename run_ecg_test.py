#!/usr/bin/env python3
"""
Quick test run for ECG with LoRA (minimal config)
Purpose: Verify setup works before full experiments
"""

import sys
import yaml

sys.path.insert(0, 'ORCA/src')

from main import main

def dict_to_munch(d):
    """Recursively convert dict to Munch"""
    from munch import Munch
    if isinstance(d, dict):
        return Munch({k: dict_to_munch(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_munch(item) for item in d]
    else:
        return d

if __name__ == '__main__':
    print("="*70)
    print("  ECG: Quick Test (1 epoch, 64 samples)")
    print("="*70)
    print()

    with open('ORCA/src/configs/ecg_test.yaml', 'r') as f:
        config = yaml.safe_load(f)
        args = dict_to_munch(config['hyperparameters'])

    main(False, args)
