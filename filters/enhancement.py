"""
Image enhancement operations:
- Histogram equalization
- Normalization
- Thresholding (Otsu)
- Color → Gray + RGB histogram
- Frequency domain filters (low-pass / high-pass)
- Hybrid images
"""
import numpy as np
import cv2
from .base import ImageFilter


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


# ──────────────────────────────────────────────
# Histogram Equalization (from scratch)
# ──────────────────────────────────────────────
class HistogramEqualizer(ImageFilter):
    """
    Equalizes a grayscale image using CDF mapping.
    Works channel-by-channel on color images.
    """

    @property
    def name(self) -> str:
        return "Histogram Equalization"

    def apply(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            # Convert to YCrCb, equalize only Y (luminance)
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = self._equalize_channel(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        return self._equalize_channel(image)

    @staticmethod
    def _equalize_channel(channel: np.ndarray) -> np.ndarray:
        hist = np.bincount(channel.flatten(), minlength=256).astype(np.float64)
        cdf = hist.cumsum()
        cdf_min = cdf[cdf > 0].min()
        total = channel.size
        lut = np.round((cdf - cdf_min) / (total - cdf_min) * 255).astype(np.uint8)
        return lut[channel]


# ──────────────────────────────────────────────
# Normalization (from scratch)
# ──────────────────────────────────────────────
class ImageNormalizer(ImageFilter):
    """Stretches pixel values to [0, 255] using min-max normalization."""

    @property
    def name(self) -> str:
        return "Image Normalization"

    def apply(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float64)
        min_val, max_val = img.min(), img.max()
        if max_val == min_val:
            return image
        normalized = (img - min_val) / (max_val - min_val) * 255
        return normalized.astype(np.uint8)


# ──────────────────────────────────────────────
# Thresholding — Otsu's method (from scratch)
# ──────────────────────────────────────────────
class OtsuThreshold(ImageFilter):
    """
    Binarizes an image using Otsu's optimal threshold.
    Computed from scratch via between-class variance maximization.
    """

    @property
    def name(self) -> str:
        return "Otsu Thresholding"

    def apply(self, image: np.ndarray) -> np.ndarray:
        gray = _to_gray(image)
        threshold = self._compute_otsu(gray)
        binary = (gray >= threshold).astype(np.uint8) * 255
        return binary

    @staticmethod
    def _compute_otsu(gray: np.ndarray) -> int:
        hist = np.bincount(gray.flatten(), minlength=256).astype(np.float64)
        total = gray.size
        prob = hist / total
        best_threshold, best_var = 0, 0.0

        for t in range(1, 256):
            w0 = prob[:t].sum()
            w1 = prob[t:].sum()
            if w0 == 0 or w1 == 0:
                continue
            mu0 = (np.arange(t) * prob[:t]).sum() / w0
            mu1 = (np.arange(t, 256) * prob[t:]).sum() / w1
            var_between = w0 * w1 * (mu0 - mu1) ** 2
            if var_between > best_var:
                best_var = var_between
                best_threshold = t

        return best_threshold


# ──────────────────────────────────────────────
# Frequency Domain Filters (from scratch via FFT)
# ──────────────────────────────────────────────
class FrequencyFilter(ImageFilter):
    """Base for frequency domain filters using numpy FFT."""

    def __init__(self, cutoff: float = 30.0, mode: str = "lowpass"):
        self._cutoff = cutoff
        self._mode = mode  # "lowpass" or "highpass"

    @property
    def name(self) -> str:
        return f"Frequency {'Low' if self._mode == 'lowpass' else 'High'}-Pass Filter"

    @property
    def description(self) -> str:
        return f"FFT-based {self._mode} filter, cutoff={self._cutoff}"

    def apply(self, image: np.ndarray) -> np.ndarray:
        gray = _to_gray(image).astype(np.float64)
        return self._apply_freq_filter(gray)

    def _apply_freq_filter(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)

        # Build circular mask
        cy, cx = h // 2, w // 2
        y_grid, x_grid = np.ogrid[:h, :w]
        dist = np.sqrt((y_grid - cy) ** 2 + (x_grid - cx) ** 2)
        mask = dist <= self._cutoff  # True inside circle

        if self._mode == "highpass":
            mask = ~mask

        fshift_filtered = fshift * mask
        f_back = np.fft.ifftshift(fshift_filtered)
        result = np.abs(np.fft.ifft2(f_back))
        return np.clip(result, 0, 255).astype(np.uint8)


class LowPassFreqFilter(FrequencyFilter):
    def __init__(self, cutoff: float = 30.0):
        super().__init__(cutoff, "lowpass")

    @property
    def name(self) -> str:
        return "Frequency Low-Pass Filter"


class HighPassFreqFilter(FrequencyFilter):
    def __init__(self, cutoff: float = 30.0):
        super().__init__(cutoff, "highpass")

    @property
    def name(self) -> str:
        return "Frequency High-Pass Filter"


# ──────────────────────────────────────────────
# Hybrid Images
# ──────────────────────────────────────────────
class HybridImageCreator:
    """
    Creates a hybrid image by blending:
      - Low-frequency content from image A
      - High-frequency content from image B
    """

    def __init__(self, low_cutoff: float = 20.0, high_cutoff: float = 20.0):
        self._low_filter = LowPassFreqFilter(low_cutoff)
        self._high_filter = HighPassFreqFilter(high_cutoff)

    def create(self, image_a: np.ndarray, image_b: np.ndarray) -> np.ndarray:
        """Both images should be the same size."""
        low = self._low_filter.apply(image_a).astype(np.float64)
        high = self._high_filter.apply(image_b).astype(np.float64)
        hybrid = (low + high) / 2
        return np.clip(hybrid, 0, 255).astype(np.uint8)
