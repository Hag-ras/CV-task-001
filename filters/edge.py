"""
Edge detection filter implementations from scratch.
Sobel, Roberts, Prewitt (manual), and Canny (opencv).
"""
import numpy as np
import cv2
from .base import ImageFilter, KernelFilter


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Convert to grayscale if needed."""
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


class GradientEdgeDetector(KernelFilter):
    """
    Base class for gradient-based edge detectors (Sobel, Roberts, Prewitt).
    Applies Gx and Gy kernels, computes magnitude = sqrt(Gx²+Gy²).
    Subclasses only need to supply the two directional kernels.
    Open/Closed: extend by subclassing, not modifying.
    """

    def __init__(self):
        """Initialize gradient edge detector."""
        super().__init__()  # CRITICAL: Initialize parent class
        self._kernel_cache_x = {}  # Separate cache for x kernels
        self._kernel_cache_y = {}  # Separate cache for y kernels

    @property
    def kernel_x(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def kernel_y(self) -> np.ndarray:
        raise NotImplementedError

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply edge detector and return magnitude image."""
        gray = _to_gray(image).astype(np.float64)
        
        # OPTIMIZATION: Use cached kernels
        gx = self._convolve_channel(gray, self.kernel_x)
        gy = self._convolve_channel(gray, self.kernel_y)
        
        # OPTIMIZATION: Use np.hypot for better numerical stability
        magnitude = np.hypot(gx, gy)
        
        return np.clip(magnitude, 0, 255).astype(np.uint8)

    def apply_directional(self, image: np.ndarray):
        """Return (gx_img, gy_img, magnitude_img) for visualization."""
        gray = _to_gray(image).astype(np.float64)
        
        # OPTIMIZATION: Use cached kernels
        gx = self._convolve_channel(gray, self.kernel_x)
        gy = self._convolve_channel(gray, self.kernel_y)
        
        # OPTIMIZATION: Use np.hypot for magnitude
        magnitude = np.hypot(gx, gy)
        
        return (
            np.clip(np.abs(gx), 0, 255).astype(np.uint8),
            np.clip(np.abs(gy), 0, 255).astype(np.uint8),
            np.clip(magnitude, 0, 255).astype(np.uint8),
        )

    def _convolve_channel(self, channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        OPTIMIZED: Vectorized convolution for edge detection.
        Edge detection doesn't need normalization.
        """
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2
        
        # Reflect padding
        padded = np.pad(channel, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        
        # OPTIMIZATION: Use sliding_window_view for vectorized convolution
        from numpy.lib.stride_tricks import sliding_window_view
        
        # Create windows view (no data copy)
        windows = sliding_window_view(padded, (kh, kw))
        
        # Reshape kernel for broadcasting
        kernel_reshaped = kernel.reshape(1, 1, kh, kw)
        
        # Vectorized convolution
        result = np.sum(windows * kernel_reshaped, axis=(2, 3))
        
        return result

    # KernelFilter requires this, but gradient detectors override apply()
    def get_kernel(self) -> np.ndarray:
        return self.kernel_x


class SobelEdge(GradientEdgeDetector):
    """Sobel edge detector with 3×3 kernels."""

    def __init__(self):
        super().__init__()  # CRITICAL: Initialize parent
        self._kernel_x = np.array([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]], dtype=np.float64)
        self._kernel_y = np.array([[-1, -2, -1],
                                   [ 0,  0,  0],
                                   [ 1,  2,  1]], dtype=np.float64)

    @property
    def name(self) -> str:
        return "Sobel Edge Detector"

    @property
    def description(self) -> str:
        return "Sobel 3×3 gradient — shows edges in X, Y, and combined"

    @property
    def kernel_x(self) -> np.ndarray:
        return self._kernel_x

    @property
    def kernel_y(self) -> np.ndarray:
        return self._kernel_y


class RobertsEdge(GradientEdgeDetector):
    """Roberts Cross edge detector with 2×2 kernels."""

    def __init__(self):
        super().__init__()  # CRITICAL: Initialize parent
        self._kernel_x = np.array([[1,  0],
                                   [0, -1]], dtype=np.float64)
        self._kernel_y = np.array([[ 0, 1],
                                   [-1, 0]], dtype=np.float64)

    @property
    def name(self) -> str:
        return "Roberts Edge Detector"

    @property
    def description(self) -> str:
        return "Roberts Cross 2×2 gradient operator"

    @property
    def kernel_x(self) -> np.ndarray:
        return self._kernel_x

    @property
    def kernel_y(self) -> np.ndarray:
        return self._kernel_y


class PrewittEdge(GradientEdgeDetector):
    """Prewitt edge detector with 3×3 kernels."""

    def __init__(self):
        super().__init__()  # CRITICAL: Initialize parent
        self._kernel_x = np.array([[-1, 0, 1],
                                   [-1, 0, 1],
                                   [-1, 0, 1]], dtype=np.float64)
        self._kernel_y = np.array([[-1, -1, -1],
                                   [ 0,  0,  0],
                                   [ 1,  1,  1]], dtype=np.float64)

    @property
    def name(self) -> str:
        return "Prewitt Edge Detector"

    @property
    def description(self) -> str:
        return "Prewitt 3×3 gradient operator"

    @property
    def kernel_x(self) -> np.ndarray:
        return self._kernel_x

    @property
    def kernel_y(self) -> np.ndarray:
        return self._kernel_y


class CannyEdge(ImageFilter):
    """
    Canny edge detector using OpenCV (as required by task spec).
    Does NOT show directional decomposition (full pipeline only).
    """

    def __init__(self, low_threshold: int = 50, high_threshold: int = 150):
        # Note: ImageFilter doesn't have __init__, so no super() needed
        self._low = low_threshold
        self._high = high_threshold

    @property
    def name(self) -> str:
        return "Canny Edge Detector"

    @property
    def description(self) -> str:
        return f"OpenCV Canny (low={self._low}, high={self._high})"

    def apply(self, image: np.ndarray) -> np.ndarray:
        gray = _to_gray(image)
        return cv2.Canny(gray, self._low, self._high)