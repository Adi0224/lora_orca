"""
LoRA integration utilities for ORCA experiments
"""
import torch
from peft import LoraConfig, get_peft_model, TaskType


def add_lora_to_model(model, rank=4, target_modules=None, lora_alpha=16, lora_dropout=0.1):
    """
    Add LoRA adapters to a model

    Args:
        model: The model to add LoRA to
        rank: LoRA rank (r parameter)
        target_modules: List of module names to apply LoRA to.
                       If None, applies to query and value projections
        lora_alpha: LoRA alpha parameter (scaling factor)
        lora_dropout: Dropout probability for LoRA layers

    Returns:
        model: Model with LoRA adapters added
        trainable_params: Number of trainable parameters
        all_params: Total number of parameters
    """

    # Default to query and value projections if not specified
    if target_modules is None:
        # For RoBERTa/BERT models
        target_modules = ["query", "value"]

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION  # Since we're using embeddings
    )

    model = get_peft_model(model, lora_config)

    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())

    print(f"LoRA Configuration:")
    print(f"  Rank: {rank}")
    print(f"  Target modules: {target_modules}")
    print(f"  Alpha: {lora_alpha}")
    print(f"  Dropout: {lora_dropout}")
    print(f"  Trainable params: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.2f}%)")

    return model, trainable_params, all_params


def print_lora_stats(model):
    """Print statistics about LoRA adapters in the model"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nLoRA Model Stats:")
    print(f"  Trainable: {trainable:,}")
    print(f"  Total: {total:,}")
    print(f"  Trainable %: {100 * trainable / total:.4f}%")
