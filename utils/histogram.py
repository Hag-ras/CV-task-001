"""
Histogram and distribution curve utilities.
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.figure
from typing import Optional


def compute_histogram(channel: np.ndarray) -> np.ndarray:
    """Compute pixel frequency histogram for a single channel (0-255)."""
    return np.bincount(channel.flatten(), minlength=256).astype(np.float64)


def compute_cdf(hist: np.ndarray) -> np.ndarray:
    """Compute normalized CDF from histogram."""
    cdf = hist.cumsum()
    return cdf / cdf[-1]


def plot_gray_histogram(image: np.ndarray) -> matplotlib.figure.Figure:
    """Plot grayscale histogram + CDF overlay."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    hist = compute_histogram(gray)
    cdf = compute_cdf(hist)

    fig, ax1 = plt.subplots(figsize=(7, 3), facecolor="#0f0f0f")
    ax2 = ax1.twinx()

    ax1.bar(range(256), hist, color="#4ade80", alpha=0.7, width=1.0)
    ax2.plot(range(256), cdf * hist.max(), color="#f97316", linewidth=2, label="CDF")

    ax1.set_facecolor("#0f0f0f")
    ax1.tick_params(colors="white")
    ax2.tick_params(colors="white")
    ax1.set_xlabel("Pixel Value", color="white")
    ax1.set_ylabel("Frequency", color="#4ade80")
    ax2.set_ylabel("CDF (scaled)", color="#f97316")
    ax1.set_title("Grayscale Histogram + CDF", color="white", fontsize=11)
    fig.tight_layout()
    return fig


def plot_rgb_histograms(image: np.ndarray) -> matplotlib.figure.Figure:
    """Plot R, G, B histograms + their CDFs."""
    if image.ndim == 2:
        return plot_gray_histogram(image)

    colors = [("B", "#60a5fa"), ("G", "#4ade80"), ("R", "#f87171")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3), facecolor="#0f0f0f")

    for idx, (label, color) in enumerate(colors):
        channel = image[:, :, idx]
        hist = compute_histogram(channel)
        cdf = compute_cdf(hist)

        ax = axes[idx]
        ax2 = ax.twinx()
        ax.bar(range(256), hist, color=color, alpha=0.6, width=1.0)
        ax2.plot(range(256), cdf * hist.max(), color="white", linewidth=1.5)
        ax.set_facecolor("#0f0f0f")
        ax.tick_params(colors="white", labelsize=7)
        ax2.tick_params(colors="white", labelsize=7)
        ax.set_title(f"Channel {label}", color=color, fontsize=10)

    fig.suptitle("RGB Histograms + CDF", color="white", fontsize=12)
    fig.tight_layout()
    return fig
