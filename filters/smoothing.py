"""
Low-pass (smoothing) filter implementations from scratch.
Average, Gaussian, and Median filters.
"""
import numpy as np
from .base import KernelFilter, ImageFilter
from numpy.lib.stride_tricks import sliding_window_view


class AverageFilter(KernelFilter):
    """Box/averaging filter using convolution."""

    def __init__(self, kernel_size: int = 3):
        # CRITICAL: Call parent constructor to initialize _kernel_cache
        super().__init__()
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
        kernel = self.get_kernel()
        # Use normalize=False since kernel already sums to 1
        return self._convolve(image, kernel, normalize=False)


class GaussianFilter(KernelFilter):
    """Gaussian blur filter with hand-crafted kernel."""

    def __init__(self, kernel_size: int = 3, sigma: float = 1.0):
        # CRITICAL: Call parent constructor to initialize _kernel_cache
        super().__init__()
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
        kernel = self.get_kernel()
        
        # For larger kernels, use separable optimization (much faster)
        if self._ksize > 5:  # Only optimize for larger kernels
            return self._apply_separable(image)
        
        return self._convolve(image, kernel, normalize=False)
    
    def _apply_separable(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian filter using separable convolution.
        This is 10-50x faster for large kernels while maintaining exact same result.
        """
        # Create 1D Gaussian kernel
        k = self._ksize
        ax = np.linspace(-(k // 2), k // 2, k)
        kernel_1d = np.exp(-(ax ** 2) / (2 * self._sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Use separable convolution (two 1D passes)
        return self._convolve_separable(image, kernel_1d, kernel_1d)


class MedianFilter(ImageFilter):
    """
    Median filter implemented from scratch.
    Uses sliding window median (not convolution — median is non-linear).
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
        """
        Optimized median filter using vectorized operations.
        Maintains exact same logic but runs much faster.
        """
        k = self._ksize
        pad = k // 2
        
        # Reflect padding (same as before)
        padded = np.pad(channel.astype(np.float64), pad, mode='reflect')
        h, w = channel.shape
        
        # ORIGINAL LOGIC (commented out):
        # for i in range(h):
        #     for j in range(w):
        #         region = padded[i:i + k, j:j + k].flatten()
        #         output[i, j] = np.median(region)
        
        # OPTIMIZED VERSION (same logic, vectorized):
        # Create sliding windows - this creates a view, no data copy
        windows = sliding_window_view(padded, (k, k))
        
        # Reshape windows to 2D array where each row is a neighborhood
        # Shape: (h, w, k, k) -> (h*w, k*k)
        windows_flat = windows.reshape(h * w, -1)
        
        # Calculate median along each row (vectorized operation)
        medians = np.median(windows_flat, axis=1)
        
        # Reshape back to image dimensions
        output = medians.reshape(h, w)
        
        return output.astype(np.uint8)
    
    def _apply_channel_original(self, channel: np.ndarray) -> np.ndarray:
        """
        Keep original implementation as reference.
        This is slower but maintained for educational purposes.
        """
        k = self._ksize
        pad = k // 2
        padded = np.pad(channel.astype(np.float64), pad, mode='reflect')
        h, w = channel.shape
        output = np.zeros((h, w), dtype=np.float64)

        for i in range(h):
            for j in range(w):
                region = padded[i:i + k, j:j + k].flatten()
                output[i, j] = np.median(region)

        return output.astype(np.uint8)