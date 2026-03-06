"""
Test case scenarios for each task implemented.
Runs all filters on a sample image and saves outputs to test_outputs/.
"""
import sys
import os

# Ensure parent directory is on path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use non-GUI backend to avoid Qt conflicts with OpenCV
import matplotlib
matplotlib.use('Agg')

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from filters import (
    UniformNoise, GaussianNoise, SaltAndPepperNoise,
    AverageFilter, GaussianFilter, MedianFilter,
    SobelEdge, RobertsEdge, PrewittEdge, CannyEdge,
    HistogramEqualizer, ImageNormalizer,
    LowPassFreqFilter, HighPassFreqFilter, HybridImageCreator,
)
from utils.histogram import plot_gray_histogram, plot_rgb_histograms

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
TEST_IMAGE1 = os.path.join(os.path.dirname(__file__), "images/sample.png")
TEST_IMAGE2 = os.path.join(os.path.dirname(__file__), "images/sample2.png")
LOW_CONTRAST= os.path.join(os.path.dirname(__file__), "images/low_contrast.png")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_outputs")


def ensure_output_dir():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def save(name: str, img: np.ndarray):
    """Save image to test_outputs/ with given name."""
    path = os.path.join(OUTPUT_DIR, name)
    cv2.imwrite(path, img)
    print(f"  ✓ Saved {name}")


def save_fig(name: str, fig: plt.Figure):
    """Save matplotlib figure to test_outputs/."""
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Saved {name}")


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


# ─────────────────────────────────────────────────────────────
# Test Scenarios
# ─────────────────────────────────────────────────────────────

def test_task1_io_grayscale(img: np.ndarray):
    """Task 1: Image I/O and Grayscale conversion."""
    print("\n[Task 1] Image I/O and Grayscale")
    save("01_original.png", img)
    gray = to_gray(img)
    save("01_grayscale.png", gray)


def test_task2_noise(img: np.ndarray):
    """Task 2: Additive Noise (Uniform, Gaussian, Salt & Pepper)."""
    print("\n[Task 2] Additive Noise")
    
    # Uniform noise
    uniform = UniformNoise(low=-50, high=50)
    save("02_noise_uniform.png", uniform.apply(img))
    
    # Gaussian noise
    gaussian = GaussianNoise(mean=0, std=25)
    save("02_noise_gaussian.png", gaussian.apply(img))
    
    # Salt & Pepper noise
    sap = SaltAndPepperNoise(density=0.05)
    save("02_noise_salt_pepper.png", sap.apply(img))


def test_task3_smoothing(img: np.ndarray):
    """Task 3: Spatial Low-Pass Filters (Average, Gaussian, Median)."""
    print("\n[Task 3] Spatial Smoothing Filters")
    
    # Add some noise first to demonstrate filtering
    noisy = GaussianNoise(mean=0, std=30).apply(img)
    save("03_input_noisy.png", noisy)
    
    # Average filter
    avg = AverageFilter(kernel_size=5)
    save("03_smooth_average.png", avg.apply(noisy))
    
    # Gaussian filter
    gauss = GaussianFilter(kernel_size=5, sigma=1.5)
    save("03_smooth_gaussian.png", gauss.apply(noisy))
    
    # Median filter (best for salt & pepper)
    sp_noisy = SaltAndPepperNoise(density=0.05).apply(img)
    save("03_input_salt_pepper.png", sp_noisy)
    median = MedianFilter(kernel_size=5)
    save("03_smooth_median.png", median.apply(sp_noisy))


def test_task4_frequency_filters(img: np.ndarray):
    """Task 4: Frequency Domain Filters (Low-Pass, High-Pass)."""
    print("\n[Task 4] Frequency Domain Filters")
    
    # Low-pass (blurs, keeps structure)
    lpf = LowPassFreqFilter(cutoff=30)
    save("04_freq_lowpass.png", lpf.apply(img))
    
    # High-pass (edges, details)
    hpf = HighPassFreqFilter(cutoff=30)
    save("04_freq_highpass.png", hpf.apply(img))


def test_task5_edge_detection(img: np.ndarray):
    """Task 5: Edge Detection (Sobel, Roberts, Prewitt, Canny)."""
    print("\n[Task 5] Edge Detection")
    
    # Sobel
    sobel = SobelEdge()
    gx, gy, mag = sobel.apply_directional(img)
    save("05_edge_sobel_gx.png", gx)
    save("05_edge_sobel_gy.png", gy)
    save("05_edge_sobel_mag.png", mag)
    
    # Roberts
    roberts = RobertsEdge()
    save("05_edge_roberts.png", roberts.apply(img))
    
    # Prewitt
    prewitt = PrewittEdge()
    save("05_edge_prewitt.png", prewitt.apply(img))
    
    # Canny (OpenCV)
    canny = CannyEdge(low_threshold=50, high_threshold=150)
    save("05_edge_canny.png", canny.apply(img))


def test_task6_7_histogram(img: np.ndarray):
    """Task 6-7: Histogram, CDF, and Equalization."""
    print("\n[Task 6-7] Histogram, CDF, Equalization")
    
    # Grayscale histogram
    fig = plot_gray_histogram(img, show_stats=True)
    save_fig("06_histogram_gray.png", fig)
    
    # RGB histograms
    fig = plot_rgb_histograms(img)
    save_fig("06_histogram_rgb.png", fig)
    
    # Histogram equalization
    eq = HistogramEqualizer()
    equalized = eq.apply(img)
    save("07_equalized.png", equalized)
    
    # Histogram after equalization
    fig = plot_gray_histogram(equalized, show_stats=True)
    save_fig("07_histogram_equalized.png", fig)


def test_task8_normalization(img: np.ndarray):
    """Task 8: Normalization."""
    print("\n[Task 8] Normalization")
    
    # # Simulate low-contrast image
    # low_contrast = (img.astype(np.float64) * 0.3 + 50).astype(np.uint8)
    save("08_low_contrast.png", img)
    
    # Normalize
    norm = ImageNormalizer()
    normalized = norm.apply(img)
    save("08_normalized.png", normalized)


# def test_task9_color_histograms(img: np.ndarray):
#     """Task 9: Color Image Analysis (per-channel histograms)."""
#     print("\n[Task 9] Color Histograms")
    
#     # Split channels manually (no cv2.split)
#     b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
#     save("09_channel_blue.png", b)
#     save("09_channel_green.png", g)
#     save("09_channel_red.png", r)
    
#     # Already saved RGB histogram in task 6-7
#     fig = plot_rgb_histograms(img)
#     save_fig("09_rgb_histograms.png", fig)


def test_task9_hybrid_images(img1: np.ndarray, img2: np.ndarray):
    """Task 9: Hybrid Images."""
    print("\n[Task 10] Hybrid Images")
    
    # Resize img2 to match img1 dimensions
    img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Create hybrid: low-freq from img1 + high-freq from img2
    hybrid_creator = HybridImageCreator(low_cutoff=25, high_cutoff=25)
    hybrid = hybrid_creator.create(img1, img2_resized)
    save("10_hybrid_input_a.png", img1)
    save("10_hybrid_input_b.png", img2_resized)
    save("10_hybrid_result.png", hybrid)


def create_comparison_grid(img: np.ndarray):
    """Create a side-by-side comparison grid of key transformations."""
    print("\n[Summary] Creating comparison grid")
    
    gray = to_gray(img)
    
    # Create subplots
    fig, axes = plt.subplots(3, 4, figsize=(16, 12), facecolor='#0f0f0f')
    fig.suptitle('Image Processing Lab – Test Results', fontsize=16, color='white', fontweight='bold')
    
    results = [
        ("Original", img),
        ("Grayscale", gray),
        ("Gaussian Noise", GaussianNoise(0, 25).apply(img)),
        ("Uniform Noise", UniformNoise(-40, 40).apply(img)),
        ("Average Filter", AverageFilter(5).apply(img)),
        ("Gaussian Filter", GaussianFilter(5, 1.5).apply(img)),
        ("Sobel Edges", SobelEdge().apply(img)),
        ("Canny Edges", CannyEdge(50, 150).apply(img)),
        ("LP Freq Filter", LowPassFreqFilter(30).apply(img)),
        ("HP Freq Filter", HighPassFreqFilter(30).apply(img)),
        ("Equalized", HistogramEqualizer().apply(img)),
        ("Normalized", ImageNormalizer().apply(img)),
    ]
    
    for ax, (title, result) in zip(axes.flat, results):
        ax.set_facecolor('#1a1a1a')
        if result.ndim == 3:
            ax.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(result, cmap='gray')
        ax.set_title(title, color='white', fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    save_fig("00_comparison_grid.png", fig)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Image Processing Lab — Test Suite")
    print("=" * 60)
    
    ensure_output_dir()
    
    # Load test image
    print(f"\nLoading test image: {TEST_IMAGE1}")
    img = load_image(TEST_IMAGE1)
    img2 = load_image(TEST_IMAGE2)
    low_contrast = load_image(LOW_CONTRAST)
    print(f"  Shape: {img.shape}, dtype: {img.dtype}")
    
    # Run all tests
    test_task1_io_grayscale(img)
    test_task2_noise(img)
    test_task3_smoothing(img)
    test_task4_frequency_filters(img)
    test_task5_edge_detection(img)
    test_task6_7_histogram(img)
    test_task8_normalization(low_contrast)
    test_task9_hybrid_images(img, img2)
    create_comparison_grid(img)
    
    print("\n" + "=" * 60)
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()