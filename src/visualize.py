"""
Visualization helpers for the loss surface.

Two key plot types:
1. 3D surface plot — shows the geometry vividly
2. 2D contour plot — easier to read off "width" of the minimum

Both use matplotlib so the output is publication-ready.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)
from matplotlib import cm


def plot_3d_surface(alphas, betas, losses, title="Loss Landscape",
                    save_path=None, clip_max=None, figsize=(10, 8)):
    """3D surface plot of the loss landscape."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    A, B = np.meshgrid(alphas, betas, indexing='ij')

    Z = losses.copy()
    if clip_max is not None:
        Z = np.clip(Z, None, clip_max)

    surf = ax.plot_surface(A, B, Z, cmap=cm.viridis, edgecolor='none',
                          alpha=0.85, antialiased=True)
    ax.set_xlabel(r'$\alpha$', fontsize=14)
    ax.set_ylabel(r'$\beta$', fontsize=14)
    ax.set_zlabel('Loss', fontsize=14)
    ax.set_title(title, fontsize=16)
    fig.colorbar(surf, shrink=0.5, aspect=10, label='Loss')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved {save_path}")
    return fig, ax


def plot_2d_contour(alphas, betas, losses, title="Loss Landscape (contour)",
                    save_path=None, levels=20, log_scale=True, figsize=(8, 7)):
    """2D contour plot — easier to read minimum width."""
    fig, ax = plt.subplots(figsize=figsize)

    A, B = np.meshgrid(alphas, betas, indexing='ij')

    Z = losses.copy()
    if log_scale:
        # Filled log-scale contour reveals the structure near the minimum
        Z = np.log(Z + 1e-10)

    cs = ax.contour(A, B, Z, levels=levels, cmap=cm.viridis, linewidths=0.8)
    csf = ax.contourf(A, B, Z, levels=levels, cmap=cm.viridis, alpha=0.6)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.2f')

    ax.set_xlabel(r'$\alpha$', fontsize=14)
    ax.set_ylabel(r'$\beta$', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.scatter([0], [0], color='red', s=50, zorder=5, label='Trained minimum')
    ax.legend()

    cbar = fig.colorbar(csf, ax=ax)
    cbar.set_label('log(Loss)' if log_scale else 'Loss', fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved {save_path}")
    return fig, ax


def plot_comparison(alphas, betas, losses_dict, title="Architecture Comparison",
                    save_path=None, log_scale=True, figsize=(14, 6)):
    """Side-by-side contour comparison of multiple landscapes."""
    n = len(losses_dict)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    A, B = np.meshgrid(alphas, betas, indexing='ij')

    # Use shared color scale for fair comparison
    all_z = np.concatenate([
        (np.log(l + 1e-10) if log_scale else l).ravel()
        for l in losses_dict.values()
    ])
    vmin, vmax = all_z.min(), all_z.max()

    for ax, (name, losses) in zip(axes, losses_dict.items()):
        Z = np.log(losses + 1e-10) if log_scale else losses
        cs = ax.contourf(A, B, Z, levels=20, cmap=cm.viridis,
                        vmin=vmin, vmax=vmax, alpha=0.7)
        ax.contour(A, B, Z, levels=20, colors='black', linewidths=0.3, alpha=0.4)
        ax.scatter([0], [0], color='red', s=40, zorder=5)
        ax.set_xlabel(r'$\alpha$', fontsize=12)
        ax.set_ylabel(r'$\beta$', fontsize=12)
        ax.set_title(name, fontsize=14)

    fig.suptitle(title, fontsize=16)
    fig.colorbar(cs, ax=axes, shrink=0.85,
                label='log(Loss)' if log_scale else 'Loss')

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved {save_path}")
    return fig
