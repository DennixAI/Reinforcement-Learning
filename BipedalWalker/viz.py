import os
from typing import List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_plots(metrics: List[Dict[str, Any]], out_dir: str = "plots") -> None:
    _ensure_dir(out_dir)
    if not metrics:
        return

    # Extract series
    updates = np.array([m["update"] for m in metrics], dtype=float)
    loss = np.array([m["loss"] for m in metrics], dtype=float)
    pol = np.array([m["policy_loss"] for m in metrics], dtype=float)
    val = np.array([m["value_loss"] for m in metrics], dtype=float)
    ent = np.array([m["entropy"] for m in metrics], dtype=float)
    fps = np.array([m["fps"] for m in metrics], dtype=float)
    eval_ret = np.array([m.get("eval_avg_return", np.nan) for m in metrics], dtype=float)

    # Main training curves
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), constrained_layout=True)

    # Losses
    ax = axes[0]
    ax.plot(updates, loss, label="total loss", color="#1f77b4")
    ax.plot(updates, pol, label="policy loss", color="#ff7f0e", alpha=0.9)
    ax.plot(updates, val, label="value loss", color="#2ca02c", alpha=0.9)
    ax.set_title("Losses vs Update")
    ax.set_xlabel("Update")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.2)
    ax.legend()

    # Entropy and FPS (twin axes)
    ax = axes[1]
    ax.plot(updates, ent, label="entropy", color="#9467bd")
    ax.set_title("Entropy and Throughput")
    ax.set_xlabel("Update")
    ax.set_ylabel("Entropy")
    ax.grid(True, alpha=0.2)

    ax2 = ax.twinx()
    ax2.plot(updates, fps, label="fps", color="#8c564b", alpha=0.6)
    ax2.set_ylabel("FPS")

    # Handle legends for twin axes
    lines, labels = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines + lines2, labels + labels2, loc="upper right")

    # Evaluation return (sparse points)
    ax = axes[2]
    # Mask NaNs to only plot evaluated points
    mask = ~np.isnan(eval_ret)
    ax.plot(updates[mask], eval_ret[mask], marker="o", linestyle="-", color="#d62728")
    ax.set_title("Evaluation Average Return")
    ax.set_xlabel("Update")
    ax.set_ylabel("Avg Return")
    ax.grid(True, alpha=0.2)

    out_path = os.path.join(out_dir, "training_curves.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    # Also save a focused eval-only plot if there are points
    if np.any(mask):
        fig2, axe = plt.subplots(1, 1, figsize=(10, 4), constrained_layout=True)
        axe.plot(updates[mask], eval_ret[mask], marker="o", linestyle="-", color="#d62728")
        axe.set_title("Evaluation Average Return")
        axe.set_xlabel("Update")
        axe.set_ylabel("Avg Return")
        axe.grid(True, alpha=0.2)
        fig2.savefig(os.path.join(out_dir, "eval_returns.png"), dpi=150)
        plt.close(fig2)

