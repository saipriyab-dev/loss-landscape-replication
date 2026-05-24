"""
Loss surface computation.

Given a trained model, two filter-normalized directions, and a dataloader,
compute the loss on a 2D grid in parameter space.

This is the main computational bottleneck of the project. Each grid point
requires a full forward pass over the evaluation set, so a 51x51 grid means
2601 forward passes. We use a smaller evaluation subset for speed.
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from filter_normalization import set_weights, perturb_weights


@torch.no_grad()
def evaluate_loss(model, dataloader, criterion, device, max_batches=None):
    """Compute average loss over (a subset of) the dataloader."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    for i, (x, y) in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)
    return total_loss / total_samples


def compute_loss_surface(
    model,
    trained_weights,
    direction1,
    direction2,
    dataloader,
    criterion,
    device,
    alpha_range=(-1.0, 1.0),
    beta_range=(-1.0, 1.0),
    n_points=21,
    max_batches=5,
):
    """
    Compute the 2D loss surface L(θ* + α·δ + β·η) over a grid.

    Parameters
    ----------
    model : nn.Module
        The model architecture. Weights will be overwritten during computation.
    trained_weights : list of torch.Tensor
        The trained parameter values θ*.
    direction1, direction2 : list of torch.Tensor
        Filter-normalized direction vectors.
    dataloader : DataLoader
        Evaluation data.
    criterion : nn.Module
        The loss function (e.g., CrossEntropyLoss).
    device : torch.device
        CPU or CUDA.
    alpha_range, beta_range : tuple
        (min, max) values for the two axes.
    n_points : int
        Grid resolution. 21x21 = 441 evaluations, manageable on a laptop.
        51x51 = 2601 evaluations, the paper's resolution.
    max_batches : int
        Number of mini-batches to evaluate at each grid point. Use a small
        number (e.g., 5) for fast prototyping; use None for final figures.

    Returns
    -------
    alphas, betas : np.ndarray
        1D arrays of grid coordinates.
    losses : np.ndarray
        2D array of shape (n_points, n_points) with the loss at each grid point.
    """
    alphas = np.linspace(alpha_range[0], alpha_range[1], n_points)
    betas = np.linspace(beta_range[0], beta_range[1], n_points)
    losses = np.zeros((n_points, n_points))

    total = n_points * n_points
    pbar = tqdm(total=total, desc="Computing loss surface")

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            # Set perturbed weights
            perturbed = perturb_weights(
                trained_weights, direction1, alpha, direction2, beta
            )
            set_weights(model, perturbed)

            # Evaluate loss
            loss = evaluate_loss(model, dataloader, criterion, device, max_batches)
            losses[i, j] = loss

            pbar.update(1)

    pbar.close()

    # Restore trained weights
    set_weights(model, trained_weights)

    return alphas, betas, losses
