"""
Histogram and distribution curve utilities.
Optimized with additional features while maintaining exact logic.
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.figure
from typing import Optional, Tuple, List, Union


def compute_histogram(channel: np.ndarray, normalize: bool = False) -> np.ndarray:
    """
    Compute pixel frequency histogram for a single channel (0-255).
    
    Args:
        channel: Input image channel (grayscale)
        normalize: If True, normalize histogram to sum to 1
        
    Returns:
        Histogram array of length 256
    """
    # np.bincount is already optimal - O(n) time, minimal memory
    hist = np.bincount(channel.flatten(), minlength=256).astype(np.float64)
    
    if normalize:
        hist = hist / hist.sum()
    
    return hist


def compute_cdf(hist: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Compute CDF from histogram.
    
    Args:
        hist: Input histogram array
        normalize: If True, normalize CDF to [0, 1]
        
    Returns:
        CDF array
    """
    cdf = hist.cumsum()
    if normalize and cdf[-1] > 0:
        cdf = cdf / cdf[-1]
    return cdf


def compute_stats(channel: np.ndarray) -> dict:
    """
    Compute statistical measures for a channel.
    
    Returns:
        Dictionary with min, max, mean, median, std
    """
    return {
        'min': channel.min(),
        'max': channel.max(),
        'mean': channel.mean(),
        'median': np.median(channel),
        'std': channel.std()
    }


def plot_gray_histogram(image: np.ndarray, 
                        show_stats: bool = True,
                        figsize: Tuple[int, int] = (7, 4)) -> matplotlib.figure.Figure:
    """
    Plot grayscale histogram + CDF overlay with optional statistics.
    
    Args:
        image: Input image (BGR or grayscale)
        show_stats: Whether to display statistics
        figsize: Figure size (width, height)
    """
    # Convert to grayscale if needed
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    
    # Compute histogram and CDF
    hist = compute_histogram(gray)
    cdf = compute_cdf(hist)
    
    # Compute statistics if requested
    stats = compute_stats(gray) if show_stats else None
    
    # Create figure with dark theme
    fig, ax1 = plt.subplots(figsize=figsize, facecolor="#0f0f0f")
    ax2 = ax1.twinx()
    
    # Plot histogram bars (optimized: use step plot for large data)
    ax1.bar(range(256), hist, color="#4ade80", alpha=0.7, width=1.0, 
            label="Histogram", edgecolor='none')
    
    # Plot CDF scaled to match histogram height for better visualization
    scaled_cdf = cdf * hist.max()
    ax2.plot(range(256), scaled_cdf, color="#f97316", linewidth=2.5, 
             label="CDF", linestyle='-')
    
    # Add reference lines for key points
    if stats:
        # Mean line
        ax1.axvline(stats['mean'], color="#60a5fa", linestyle='--', 
                   alpha=0.8, linewidth=1, label=f"Mean: {stats['mean']:.1f}")
        
        # Median line
        ax1.axvline(stats['median'], color="#f87171", linestyle=':', 
                   alpha=0.8, linewidth=1, label=f"Median: {stats['median']:.1f}")
    
    # Styling
    ax1.set_facecolor("#0f0f0f")
    ax1.tick_params(colors="white", labelsize=8)
    ax2.tick_params(colors="white", labelsize=8)
    
    # Set labels with colors
    ax1.set_xlabel("Pixel Value", color="white", fontsize=10)
    ax1.set_ylabel("Frequency", color="#4ade80", fontsize=10)
    ax2.set_ylabel("CDF (scaled)", color="#f97316", fontsize=10)
    
    # Title with stats
    title = "Grayscale Histogram + CDF"
    if stats:
        title += f"\nμ={stats['mean']:.1f} σ={stats['std']:.1f} | Range: [{stats['min']}, {stats['max']}]"
    ax1.set_title(title, color="white", fontsize=11, fontweight='bold')
    
    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
               facecolor="#1a1a26", edgecolor='none', labelcolor='white')
    
    fig.tight_layout()
    return fig


def plot_rgb_histograms(image: np.ndarray, 
                        show_stats: bool = True,
                        figsize: Tuple[int, int] = (15, 4)) -> matplotlib.figure.Figure:
    """
    Plot R, G, B histograms + their CDFs with optional statistics.
    
    Args:
        image: Input BGR image
        show_stats: Whether to display statistics
        figsize: Figure size (width, height)
    """
    if image.ndim == 2:
        return plot_gray_histogram(image)
    
    # Define colors for BGR channels
    colors = [
        ("B", "#60a5fa", "#3b82f6"),  # Blue: light blue for bars, darker for stats
        ("G", "#4ade80", "#22c55e"),  # Green: light green for bars, darker for stats
        ("R", "#f87171", "#ef4444")   # Red: light red for bars, darker for stats
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=figsize, facecolor="#0f0f0f")
    
    for idx, (label, bar_color, line_color) in enumerate(colors):
        channel = image[:, :, idx]
        
        # Compute histogram and CDF
        hist = compute_histogram(channel)
        cdf = compute_cdf(hist)
        stats = compute_stats(channel) if show_stats else None
        
        ax = axes[idx]
        ax2 = ax.twinx()
        
        # Plot histogram
        ax.bar(range(256), hist, color=bar_color, alpha=0.6, width=1.0, 
               edgecolor='none')
        
        # Plot CDF scaled to histogram
        scaled_cdf = cdf * hist.max()
        ax2.plot(range(256), scaled_cdf, color="white", linewidth=2, 
                label="CDF", linestyle='-')
        
        # Add reference lines for key points
        if stats:
            # Mean line
            ax.axvline(stats['mean'], color=line_color, linestyle='--', 
                      alpha=0.8, linewidth=1.5)
            
            # Add text box with stats
            stats_text = f"μ={stats['mean']:.1f}\nσ={stats['std']:.1f}"
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                   color="white", fontsize=7, verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(boxstyle="round", facecolor="#1a1a26", 
                           alpha=0.8, edgecolor=line_color))
        
        # Styling
        ax.set_facecolor("#0f0f0f")
        ax.tick_params(colors="white", labelsize=7)
        ax2.tick_params(colors="white", labelsize=7)
        
        # Title with channel and range
        ax.set_title(f"Channel {label}\n[{stats['min']}, {stats['max']}]" if stats 
                    else f"Channel {label}", 
                    color=bar_color, fontsize=10, fontweight='bold')
        
        # Set x-axis label only for middle plot
        if idx == 1:
            ax.set_xlabel("Pixel Value", color="white", fontsize=9)
    
    fig.suptitle("RGB Histograms + CDF", color="white", fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_comparison_histograms(original: np.ndarray, 
                              processed: np.ndarray,
                              title1: str = "Original",
                              title2: str = "Processed") -> matplotlib.figure.Figure:
    """
    Plot side-by-side histogram comparison.
    Useful for comparing before/after equalization or normalization.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="#0f0f0f")
    
    for idx, (img, title) in enumerate([(original, title1), (processed, title2)]):
        # Convert to grayscale if needed
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        hist = compute_histogram(gray)
        
        ax = axes[idx]
        ax.bar(range(256), hist, color="#4ade80", alpha=0.7, width=1.0)
        ax.set_facecolor("#0f0f0f")
        ax.tick_params(colors="white", labelsize=8)
        ax.set_title(title, color="white", fontsize=11)
        ax.set_xlabel("Pixel Value", color="white", fontsize=9)
        ax.set_ylabel("Frequency", color="white", fontsize=9)
    
    fig.tight_layout()
    return fig


def plot_cumulative_histograms(image: np.ndarray) -> matplotlib.figure.Figure:
    """
    Plot cumulative histograms (CDFs) for all channels.
    Useful for comparing distributions.
    """
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0f0f0f")
    
    colors = [("#60a5fa", "Blue"), ("#4ade80", "Green"), ("#f87171", "Red")]
    
    if image.ndim == 3:
        for idx, (color, label) in enumerate(colors):
            channel = image[:, :, idx]
            hist = compute_histogram(channel)
            cdf = compute_cdf(hist)
            ax.plot(range(256), cdf, color=color, linewidth=2, label=f"{label} CDF")
    else:
        hist = compute_histogram(image)
        cdf = compute_cdf(hist)
        ax.plot(range(256), cdf, color="#4ade80", linewidth=2, label="Grayscale CDF")
    
    ax.set_facecolor("#0f0f0f")
    ax.tick_params(colors="white")
    ax.set_xlabel("Pixel Value", color="white")
    ax.set_ylabel("Cumulative Probability", color="white")
    ax.set_title("Cumulative Distribution Functions (CDF)", color="white", fontsize=12)
    ax.legend(facecolor="#1a1a26", edgecolor='none', labelcolor='white')
    ax.grid(True, alpha=0.2)
    
    fig.tight_layout()
    return fig