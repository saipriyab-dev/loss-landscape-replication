"""
Filter normalization — the mathematical innovation of Li et al. (2018).

The core idea
-------------
To visualize a loss surface near a trained point θ*, we sample two random
direction vectors δ and η, and plot L(θ* + α·δ + β·η) for a grid of (α, β).

The problem: the SCALE of these directions affects the visualization. A larger
direction makes the loss change faster, making the landscape look "sharper."
This means you can't compare two networks fairly — a network with larger
weights will always look like it has a sharper landscape, even if the
underlying geometry is identical.

The solution: filter normalization. For each convolutional filter in θ*, we
scale the corresponding component of the random direction so that it has the
SAME NORM as the filter itself. This way, "stepping by amount α" means the
same thing across architectures.

Mathematically: if filter f_ij has weights θ_ij, we set
    δ_ij ← δ_ij × (||θ_ij|| / ||δ_ij||)

This is essentially per-filter normalization to a fixed scale.

Reference: Section 4 of Li et al. (2018).
"""

import torch
import torch.nn as nn


def get_weights(model):
    """Extract all parameter tensors from a model as a flat list."""
    return [p.data.clone() for p in model.parameters()]


def get_random_direction(model, seed=None):
    """Sample a random direction with the same shape as the model's parameters."""
    if seed is not None:
        torch.manual_seed(seed)

    direction = []
    for p in model.parameters():
        d = torch.randn_like(p.data)
        direction.append(d)
    return direction


def filter_normalize(direction, weights, ignore_biasbn=True):
    """
    Apply filter normalization to a direction vector.

    For each convolutional filter (4D tensor: out_channels, in_channels, H, W),
    we normalize each filter slice independently. For fully connected layers
    (2D tensor), we normalize each row.

    Parameters
    ----------
    direction : list of torch.Tensor
        The random direction vector, same shape as model parameters.
    weights : list of torch.Tensor
        The trained parameters θ* of the model.
    ignore_biasbn : bool
        If True, set biases and batch-norm parameters in the direction to zero.
        This is what the paper recommends, because biases and BN params don't
        have a meaningful "filter" structure.

    Returns
    -------
    list of torch.Tensor
        The filter-normalized direction.
    """
    normalized = []

    for d, w in zip(direction, weights):
        if w.dim() == 4:
            # Conv layer: shape (out_channels, in_channels, H, W)
            # Normalize each filter (the inner 3D tensor for each out_channel)
            d_norm = d.clone()
            for i in range(w.size(0)):
                w_norm = w[i].norm()
                d_unit_norm = d[i].norm()
                if d_unit_norm > 0:
                    d_norm[i] = d[i] * (w_norm / d_unit_norm)
            normalized.append(d_norm)
        elif w.dim() == 2:
            # Linear layer: shape (out_features, in_features)
            # Normalize each row
            d_norm = d.clone()
            for i in range(w.size(0)):
                w_norm = w[i].norm()
                d_unit_norm = d[i].norm()
                if d_unit_norm > 0:
                    d_norm[i] = d[i] * (w_norm / d_unit_norm)
            normalized.append(d_norm)
        else:
            # Bias or BN params — zero out if requested
            if ignore_biasbn:
                normalized.append(torch.zeros_like(d))
            else:
                # Scale by overall norm
                w_norm = w.norm()
                d_norm = d * (w_norm / (d.norm() + 1e-10))
                normalized.append(d_norm)

    return normalized


def set_weights(model, weights):
    """Set the model's parameters from a list of tensors."""
    for p, w in zip(model.parameters(), weights):
        p.data.copy_(w)


def perturb_weights(weights, direction1, alpha, direction2=None, beta=0.0):
    """Compute θ* + α·δ + β·η (for 2D: provide direction2 and beta)."""
    perturbed = []
    for i, w in enumerate(weights):
        new_w = w + alpha * direction1[i]
        if direction2 is not None:
            new_w = new_w + beta * direction2[i]
        perturbed.append(new_w)
    return perturbed


if __name__ == "__main__":
    # Quick sanity check
    from models import resnet20

    model = resnet20()
    weights = get_weights(model)
    direction = get_random_direction(model, seed=42)
    normalized = filter_normalize(direction, weights)

    print(f"Number of parameter tensors: {len(weights)}")
    print(f"Total parameters: {sum(w.numel() for w in weights):,}")
    print("First few tensor shapes:")
    for i in range(min(5, len(weights))):
        print(f"  weight {i}: {tuple(weights[i].shape)}")
