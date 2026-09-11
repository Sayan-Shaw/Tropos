"""
models/lora_utils.py
====================

Minimal LoRA implementation. Injects trainable low-rank matrices
into nn.Linear layers without changing frozen weights:

    output = W_frozen @ x + scale * (B @ A) @ x

where A: (rank, in_features), B: (out_features, rank).
scale = alpha / rank.

Only ~0.1–0.5% additional parameters relative to the mask decoder.
"""

import torch
import torch.nn as nn
from collections import OrderedDict


class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with LoRA adapters.
    The original weight is frozen; only A, B are trainable.
    """

    def __init__(self, original_linear, rank=4, alpha=8.0, dropout=0.1):
        super().__init__()
        self.original = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        in_f = original_linear.in_features
        out_f = original_linear.out_features

        self.lora_A = nn.Parameter(torch.randn(rank, in_f) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Freeze original
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        # Original path (frozen)
        out = self.original(x)
        # LoRA path (trainable)
        lora_out = self.lora_dropout(x) @ self.lora_A.to(x.device).T @ self.lora_B.to(x.device).T
        return out + self.scale * lora_out

    @property
    def weight(self):
        """For compatibility with code that reads .weight"""
        return self.original.weight

    @property
    def bias(self):
        return self.original.bias


def inject_lora(module, rank=4, alpha=8.0, dropout=0.1,
                target_modules=None):
    """
    Replace all nn.Linear layers in `module` with LoRALinear.

    Parameters
    ----------
    module : nn.Module — typically the mask decoder
    rank : int — LoRA rank (4 or 8 recommended)
    alpha : float — LoRA scaling factor
    dropout : float — LoRA dropout
    target_modules : set of str or None
        If given, only replace modules whose name contains one of these.
        e.g. {"q_proj", "v_proj", "out_proj", "mlp"}

    Returns
    -------
    num_replaced : int — count of layers replaced
    total_lora_params : int — total trainable parameters added
    """
    replaced = 0
    total_params = 0

    for name, child in list(module.named_modules()):
        if not isinstance(child, nn.Linear):
            continue
        if target_modules and not any(t in name for t in target_modules):
            continue

        # Navigate to parent to do the replacement
        parts = name.split('.')
        parent = module
        for part in parts[:-1]:
            parent = getattr(parent, part)

        lora = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, parts[-1], lora)
        replaced += 1
        total_params += lora.lora_A.numel() + lora.lora_B.numel()

    return replaced, total_params


def extract_lora_state_dict(module):
    """
    Extract only the LoRA parameters from a module's state dict.
    Returns an OrderedDict with keys like 'decoder.layers.0.lora_A'.
    """
    lora_sd = OrderedDict()
    for name, param in module.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            lora_sd[name] = param.data.clone()
    return lora_sd


def count_trainable_params(module):
    """Count parameters with requires_grad=True."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def count_frozen_params(module):
    """Count parameters with requires_grad=False."""
    return sum(p.numel() for p in module.parameters() if not p.requires_grad)
