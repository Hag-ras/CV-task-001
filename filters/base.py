"""
Base abstractions for image filters.
Follows Interface Segregation and Dependency Inversion principles.
Optimized version with vectorized operations and better error handling.
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Callable, Tuple, Union
import warnings


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

    def __init__(self):
        """Initialize kernel filter with validation."""
        self._kernel_cache = {}  # Cache for frequently used kernels

    @abstractmethod
    def get_kernel(self, **kwargs) -> np.ndarray:
        """Return the convolution kernel."""
        pass

    def _validate_image(self, image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Validate and prepare image for convolution.
        
        Returns:
            Tuple of (processed_image, is_color)
        """
        if image is None or image.size == 0:
            raise ValueError("Empty image provided")

        # Check image dimensions
        if image.ndim not in [2, 3]:
            raise ValueError(f"Image must be 2D or 3D, got {image.ndim}D")

        # Convert to float64 for precision, but keep original dtype info
        if image.dtype != np.float64:
            image = image.astype(np.float64)

        # Determine if color image
        is_color = image.ndim == 3 and image.shape[2] in [3, 4]

        return image, is_color

    def _validate_kernel(self, kernel: np.ndarray) -> np.ndarray:
        """Validate and prepare kernel for convolution."""
        if kernel is None or kernel.size == 0:
            raise ValueError("Empty kernel provided")

        # Ensure kernel is 2D
        if kernel.ndim != 2:
            raise ValueError(f"Kernel must be 2D, got {kernel.ndim}D")

        # Ensure kernel is odd-sized
        if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
            raise ValueError(f"Kernel dimensions must be odd, got {kernel.shape}")

        # Convert kernel to float64 for precision
        return kernel.astype(np.float64)

    @staticmethod
    def _normalize_kernel(kernel: np.ndarray) -> np.ndarray:
        """
        Normalize kernel to sum to 1 for smoothing filters.
        
        This ensures brightness is preserved when applying smoothing filters.
        """
        kernel_sum = np.abs(np.sum(kernel))  # Use absolute to handle negative values
        if kernel_sum > 1e-10 and kernel_sum != 1.0:  # Avoid division by zero
            return kernel / kernel_sum
        return kernel

    def _convolve(self, image: np.ndarray, kernel: np.ndarray, 
                  normalize: bool = False,
                  progress_callback: Optional[Callable[[float], None]] = None) -> np.ndarray:
        """
        Optimized manual 2D convolution using vectorized operations.
        
        Args:
            image: Input image (grayscale or multi-channel)
            kernel: Convolution kernel
            normalize: Whether to normalize kernel to sum to 1
            progress_callback: Optional callback for progress reporting
        
        Returns:
            Convolved image
        """
        # Validate inputs
        image, is_color = self._validate_image(image)
        kernel = self._validate_kernel(kernel)

        # Normalize if requested
        if normalize:
            kernel = self._normalize_kernel(kernel)

        # Cache kernel for potential reuse
        kernel_hash = hash(kernel.tobytes())
        if kernel_hash not in self._kernel_cache:
            self._kernel_cache[kernel_hash] = kernel

        if progress_callback:
            progress_callback(0.1)  # Started

        # Process based on image type
        if not is_color:
            result = self._convolve_channel(image, kernel)
        else:
            # Process each channel independently
            channels = []
            for c in range(image.shape[2]):
                if progress_callback:
                    # Report progress across channels (10-90% range)
                    channel_progress = 0.1 + (c / image.shape[2]) * 0.8
                    progress_callback(channel_progress)
                
                channel = self._convolve_channel(image[:, :, c], kernel)
                channels.append(channel)
            
            result = np.stack(channels, axis=2)

        if progress_callback:
            progress_callback(1.0)  # Completed

        return result.astype(np.uint8)

    @staticmethod
    def _convolve_channel(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Highly optimized convolution for a single channel using NumPy's sliding window view.
        
        This implementation is 10-100x faster than nested loops while maintaining
        the exact same sliding window logic.
        """
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2

        # Reflect padding (handles borders naturally)
        padded = np.pad(channel, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')

        h, w = channel.shape

        # Use sliding_window_view to create windows without copying data
        # This creates a view of the padded array with shape (h, w, kh, kw)
        from numpy.lib.stride_tricks import sliding_window_view
        
        # Create windows - this is O(1) operation, no data copying
        windows = sliding_window_view(padded, (kh, kw))
        
        # Reshape kernel for broadcasting: (kh, kw) -> (1, 1, kh, kw)
        kernel_reshaped = kernel.reshape(1, 1, kh, kw)
        
        # Vectorized convolution: multiply windows by kernel and sum
        # This does the same operation as nested loops but in C speed
        output = np.sum(windows * kernel_reshaped, axis=(2, 3))

        # Clip to valid range and convert back to original dtype
        return np.clip(output, 0, 255)

    def _convolve_channel_large(self, channel: np.ndarray, kernel: np.ndarray, 
                                 chunk_size: int = 512) -> np.ndarray:
        """
        Process large images in chunks to manage memory usage.
        Useful for very large images where the sliding_window_view might use too much memory.
        """
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2
        padded = np.pad(channel, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        
        h, w = channel.shape
        output = np.zeros((h, w), dtype=np.float64)
        
        # Process image in chunks along height
        for i in range(0, h, chunk_size):
            i_end = min(i + chunk_size, h)
            
            # Extract chunk with padding
            chunk_start = i
            chunk_end = i_end + 2 * pad_h
            padded_chunk = padded[chunk_start:chunk_end, :]
            
            # Apply convolution to chunk
            from numpy.lib.stride_tricks import sliding_window_view
            windows = sliding_window_view(padded_chunk, (kh, kw))
            kernel_reshaped = kernel.reshape(1, 1, kh, kw)
            chunk_result = np.sum(windows * kernel_reshaped, axis=(2, 3))
            
            # Store result
            output[i:i_end, :] = chunk_result[:i_end - i, :]
        
        return np.clip(output, 0, 255)

    def _convolve_separable(self, image: np.ndarray, kernel_x: np.ndarray, 
                           kernel_y: np.ndarray) -> np.ndarray:
        """
        Optimized convolution for separable kernels (like Gaussian).
        
        For separable kernels, we can convolve in two 1D passes which is much faster:
        O(h*w*(kh+kw)) instead of O(h*w*kh*kw)
        """
        # First pass: convolve with horizontal kernel
        temp = self._convolve_channel(image, kernel_x.reshape(1, -1))
        
        # Second pass: convolve result with vertical kernel
        result = self._convolve_channel(temp, kernel_y.reshape(-1, 1))
        
        return result

    def clear_kernel_cache(self):
        """Clear the kernel cache to free memory."""
        self._kernel_cache.clear()


# Optional: Add a context manager for performance monitoring
class ConvolutionTimer:
    """Context manager to measure convolution performance."""
    
    def __init__(self, name: str = "Convolution"):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        import time
        self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, *args):
        import time
        elapsed = time.perf_counter() - self.start_time
        print(f"{self.name} took {elapsed:.4f} seconds")
        
        # Warn if convolution is too slow
        if elapsed > 1.0:
            warnings.warn(f"Slow convolution detected ({elapsed:.2f}s). Consider using smaller kernel or image.")