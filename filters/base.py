"""
Base abstractions for image filters.
Convolution implemented from scratch using NumPy's sliding_window_view.
"""
from abc import ABC, abstractmethod
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from typing import Optional, Callable, Tuple


class ImageFilter(ABC):
    """Abstract base class for all image filters."""

    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply the filter to an image and return the result."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the filter."""
        pass

    @property
    def description(self) -> str:
        """Optional description for UI display."""
        return ""


class KernelFilter(ImageFilter):
    """Base class for kernel-based (convolution) filters."""

    @abstractmethod
    def get_kernel(self, **kwargs) -> np.ndarray:
        """Return the convolution kernel."""
        pass

    @staticmethod
    def _validate_image(image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Validate and prepare image. Returns (float64_image, is_color)."""
        if image is None or image.size == 0:
            raise ValueError("Empty image provided")
        if image.ndim not in [2, 3]:
            raise ValueError(f"Image must be 2D or 3D, got {image.ndim}D")

        if image.dtype != np.float64:
            image = image.astype(np.float64)
        is_color = image.ndim == 3 and image.shape[2] in [3, 4]
        return image, is_color

    @staticmethod
    def _validate_kernel(kernel: np.ndarray) -> np.ndarray:
        """Validate kernel and convert to float64."""
        if kernel is None or kernel.size == 0:
            raise ValueError("Empty kernel provided")
        if kernel.ndim != 2:
            raise ValueError(f"Kernel must be 2D, got {kernel.ndim}D")
        if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
            raise ValueError(f"Kernel dimensions must be odd, got {kernel.shape}")
        return kernel.astype(np.float64)

    @staticmethod
    def _normalize_kernel(kernel: np.ndarray) -> np.ndarray:
        """Normalize kernel to sum to 1 (preserves brightness for smoothing)."""
        kernel_sum = np.abs(np.sum(kernel))
        if kernel_sum > 1e-10 and kernel_sum != 1.0:
            return kernel / kernel_sum
        return kernel

    def _convolve(self, image: np.ndarray, kernel: np.ndarray,
                  normalize: bool = False,
                  progress_callback: Optional[Callable[[float], None]] = None) -> np.ndarray:
        """
        Manual 2D convolution using sliding_window_view.
        Handles grayscale and multi-channel images with reflect padding.
        """
        image, is_color = self._validate_image(image)
        kernel = self._validate_kernel(kernel)
        if normalize:
            kernel = self._normalize_kernel(kernel)

        if progress_callback:
            progress_callback(0.1)

        if not is_color:
            result = self._convolve_channel(image, kernel)
        else:
            channels = [self._convolve_channel(image[:, :, c], kernel)
                        for c in range(image.shape[2])]
            result = np.stack(channels, axis=2)

        if progress_callback:
            progress_callback(1.0)

        return result.astype(np.uint8)

    @staticmethod
    def _convolve_channel(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Vectorised convolution for a single channel via sliding_window_view."""
        kh, kw = kernel.shape
        pad_top, pad_bot = (kh - 1) // 2, kh // 2
        pad_left, pad_right = (kw - 1) // 2, kw // 2

        padded = np.pad(channel, ((pad_top, pad_bot), (pad_left, pad_right)), mode='reflect')
        windows = sliding_window_view(padded, (kh, kw))          # view, no copy
        output = np.sum(windows * kernel.reshape(1, 1, kh, kw),  # broadcast
                        axis=(2, 3))
        return np.clip(output, 0, 255)