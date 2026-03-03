"""
Low-pass (smoothing) filter implementations from scratch.
Average, Gaussian, and Median filters.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from .base import KernelFilter, ImageFilter


class AverageFilter(KernelFilter):
    """Box/averaging filter using convolution."""

    def __init__(self, kernel_size: int = 3):
        self._ksize = kernel_size

    @property
    def name(self) -> str:
        return f"Average Filter ({self._ksize}×{self._ksize})"

    @property
    def description(self) -> str:
        return f"Box blur with {self._ksize}×{self._ksize} kernel"

    def get_kernel(self) -> np.ndarray:
        k = self._ksize
        return np.ones((k, k), dtype=np.float64) / (k * k)

    def apply(self, image: np.ndarray) -> np.ndarray:
        return self._convolve(image, self.get_kernel())


class GaussianFilter(KernelFilter):
    """Gaussian blur filter with hand-crafted kernel."""

    def __init__(self, kernel_size: int = 3, sigma: float = 1.0):
        self._ksize = kernel_size
        self._sigma = sigma

    @property
    def name(self) -> str:
        return f"Gaussian Filter ({self._ksize}×{self._ksize}, σ={self._sigma})"

    @property
    def description(self) -> str:
        return f"Gaussian blur {self._ksize}×{self._ksize}, σ={self._sigma}"

    def get_kernel(self) -> np.ndarray:
        k = self._ksize
        ax = np.linspace(-(k // 2), k // 2, k)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * self._sigma ** 2))
        return kernel / kernel.sum()

    def apply(self, image: np.ndarray) -> np.ndarray:
        return self._convolve(image, self.get_kernel())


class MedianFilter(ImageFilter):
    """
    Median filter implemented from scratch.
    Uses sliding window median (non-linear, not convolution).
    """

    def __init__(self, kernel_size: int = 3):
        self._ksize = kernel_size

    @property
    def name(self) -> str:
        return f"Median Filter ({self._ksize}×{self._ksize})"

    @property
    def description(self) -> str:
        return f"Median filter with {self._ksize}×{self._ksize} window"

    def apply(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return self._apply_channel(image)
        channels = [self._apply_channel(image[:, :, c]) for c in range(image.shape[2])]
        return np.stack(channels, axis=2)

    def _apply_channel(self, channel: np.ndarray) -> np.ndarray:
        """Vectorised median filter using sliding_window_view."""
        k = self._ksize
        pad = k // 2
        padded = np.pad(channel.astype(np.float64), pad, mode='reflect')
        h, w = channel.shape

        windows = sliding_window_view(padded, (k, k))
        medians = np.median(windows.reshape(h * w, -1), axis=1)
        return medians.reshape(h, w).astype(np.uint8)