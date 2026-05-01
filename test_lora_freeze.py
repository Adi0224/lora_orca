#!/usr/bin/env python3
"""
Test that LoRA freeze behavior is correct:
- Base RoBERTa weights should be FROZEN in Stage 3
- Only LoRA params, embedder, and predictor should be trainable
"""

import sys
import os

_repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_repo_root, 'ORCA', 'src'))
sys.path.insert(0, os.path.join(_repo_root, 'ORCA', 'src', 'otdd'))

import torch
from munch import Munch
from embedder import wrapper1D
from task_configs import get_optimizer_scheduler

def make_args(**overrides):
    defaults = dict(
        dataset='ECG', embedder_dataset='text', objective='l2',
        weight='roberta', maxsamples=64, experiment_id='test',
        seed=0, epochs=1, embedder_epochs=0, predictor_epochs=0,
        finetune_method='all', drop_out=0, target_seq_len=64,
        use_lora=True, lora_rank=2, lora_alpha=16, lora_dropout=0.1,
        batch_size=4, eval_batch_size=100, accum=2, clip=-1,
        validation_freq=1, num_workers=0, reproducibility=False,
        valid_split=False, activation=None, device='cpu',
        optimizer=Munch(name='SGD', params=Munch(lr=0.000001, betas=[0.9, 0.98], weight_decay=0.1, momentum=0.99)),
        scheduler=Munch(name='WarmupLR', params=Munch(warmup_epochs=1, decay_epochs=10, sched=[5], base=0.2)),
        no_warmup_scheduler=Munch(name='StepLR', params=Munch(warmup_epochs=1, decay_epochs=10, sched=[5], base=0.2)),
        lr_sched_iter=False, infer_label=False,
    )
    defaults.update(overrides)
    return Munch(defaults)


def test_lora_freeze():
    """Test that Stage 3 only trains LoRA + embedder + predictor when use_lora=True"""
    print("=" * 60)
    print("TEST: LoRA parameter freezing in Stage 3")
    print("=" * 60)

    sample_shape = (1, 1, 3000)
    num_classes = 4

    model = wrapper1D(sample_shape, num_classes, weight='roberta',
                      train_epoch=0, target_seq_len=64,
                      use_lora=True, lora_rank=2, lora_alpha=16, lora_dropout=0.1)

    args = make_args(use_lora=True)

    # Simulate Stage 3: module=None triggers the fix
    args, model, optimizer, scheduler = get_optimizer_scheduler(args, model, module=None, n_train=1)

    trainable_params = []
    frozen_params = []
    trainable_count = 0
    frozen_count = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
            trainable_count += param.numel()
        else:
            frozen_params.append(name)
            frozen_count += param.numel()

    print(f"\nTrainable parameters: {trainable_count:,}")
    print(f"Frozen parameters:   {frozen_count:,}")
    print(f"Total parameters:    {trainable_count + frozen_count:,}")
    print(f"Trainable %:         {100 * trainable_count / (trainable_count + frozen_count):.2f}%")

    # Check trainable params are only lora, embedder, predictor
    errors = []
    for name in trainable_params:
        is_lora = 'lora_' in name
        is_embedder = name.startswith('embedder.')
        is_predictor = name.startswith('predictor.')
        if not (is_lora or is_embedder or is_predictor):
            errors.append(f"  UNEXPECTED trainable: {name}")

    # Check base model weights are frozen
    for name in trainable_params:
        if 'base_layer' in name:
            errors.append(f"  BASE LAYER trainable (should be frozen): {name}")

    print("\n--- Trainable parameter groups ---")
    lora_names = [n for n in trainable_params if 'lora_' in n]
    embedder_names = [n for n in trainable_params if n.startswith('embedder.')]
    predictor_names = [n for n in trainable_params if n.startswith('predictor.')]
    print(f"  LoRA params:      {len(lora_names)}")
    print(f"  Embedder params:  {len(embedder_names)}")
    print(f"  Predictor params: {len(predictor_names)}")

    if errors:
        print("\n*** FAILURES ***")
        for e in errors:
            print(e)
        print("\nTEST FAILED")
        return False
    else:
        print("\nAll trainable params are LoRA, embedder, or predictor.")
        print("All base RoBERTa weights are frozen.")
        print("TEST PASSED")
        return True


def test_no_lora_full_finetune():
    """Test that without LoRA, all params are trainable (original behavior)"""
    print("\n" + "=" * 60)
    print("TEST: Full fine-tune (no LoRA) still works")
    print("=" * 60)

    sample_shape = (1, 1, 3000)
    num_classes = 4

    model = wrapper1D(sample_shape, num_classes, weight='roberta',
                      train_epoch=0, target_seq_len=64,
                      use_lora=False)

    args = make_args(use_lora=False)
    args, model, optimizer, scheduler = get_optimizer_scheduler(args, model, module=None, n_train=1)

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())

    print(f"\nTrainable: {trainable_count:,} / {total_count:,}")

    if trainable_count == total_count:
        print("All parameters trainable (expected for full fine-tune).")
        print("TEST PASSED")
        return True
    else:
        print(f"Only {trainable_count} of {total_count} trainable - unexpected.")
        print("TEST FAILED")
        return False


if __name__ == '__main__':
    passed = 0
    failed = 0

    if test_lora_freeze():
        passed += 1
    else:
        failed += 1

    if test_no_lora_full_finetune():
        passed += 1
    else:
        failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
