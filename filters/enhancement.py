"""
Image enhancement operations:
- Histogram equalization  (CDF mapping, from scratch)
- Normalization           (min-max stretching)
- Frequency domain filters (low-pass / high-pass via FFT)
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
# Histogram Equalization
# ──────────────────────────────────────────────
class HistogramEqualizer(ImageFilter):
    """Equalizes using CDF-based LUT mapping. Color images use YCrCb luminance."""

    @property
    def name(self) -> str:
        return "Histogram Equalization"

    def apply(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
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
# Normalization
# ──────────────────────────────────────────────
class ImageNormalizer(ImageFilter):
    """Stretches pixel values to [0, 255] using min-max normalization."""

    @property
    def name(self) -> str:
        return "Image Normalization"

    def apply(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float64)
        lo, hi = img.min(), img.max()
        if hi == lo:
            return image
        return ((img - lo) / (hi - lo) * 255).astype(np.uint8)




# ──────────────────────────────────────────────
# Frequency Domain Filters (FFT)
# ──────────────────────────────────────────────
class FrequencyFilter(ImageFilter):
    """Base for frequency domain filters using numpy FFT."""

    def __init__(self, cutoff: float = 30.0, mode: str = "lowpass"):
        self._cutoff = cutoff
        self._mode = mode

    @property
    def name(self) -> str:
        return f"Frequency {'Low' if self._mode == 'lowpass' else 'High'}-Pass Filter"

    @property
    def description(self) -> str:
        return f"FFT-based {self._mode} filter, cutoff={self._cutoff}"

    def apply(self, image: np.ndarray) -> np.ndarray:
        gray = _to_gray(image).astype(np.float64)
        h, w = gray.shape

        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)

        # Circular mask
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        mask = np.sqrt((y - cy) ** 2 + (x - cx) ** 2) <= self._cutoff
        if self._mode == "highpass":
            mask = ~mask

        result = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift * mask)))
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
    Blends low-frequency content from image A with
    high-frequency content from image B.
    """

    def __init__(self, low_cutoff: float = 20.0, high_cutoff: float = 20.0):
        self._low_filter = LowPassFreqFilter(low_cutoff)
        self._high_filter = HighPassFreqFilter(high_cutoff)

    def create(self, image_a: np.ndarray, image_b: np.ndarray) -> np.ndarray:
        """Both images must be the same size."""
        low = self._low_filter.apply(image_a).astype(np.float64)
        high = self._high_filter.apply(image_b).astype(np.float64)
        return np.clip((low + high) / 2, 0, 255).astype(np.uint8)