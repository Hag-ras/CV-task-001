"""
Additive noise implementations.
Each noise type is its own class (Single Responsibility Principle).
"""
import numpy as np
from .base import ImageFilter


class UniformNoise(ImageFilter):
    """Adds uniform random noise to an image."""

    def __init__(self, low: int = -50, high: int = 50):
        self._low = low
        self._high = high

    @property
    def name(self) -> str:
        return "Uniform Noise"

    @property
    def description(self) -> str:
        return f"Uniform noise in range [{self._low}, {self._high}]"

    def apply(self, image: np.ndarray) -> np.ndarray:
        noise = np.random.uniform(self._low, self._high, image.shape).astype(np.float64)
        noisy = image.astype(np.float64) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)


class GaussianNoise(ImageFilter):
    """Adds Gaussian (normal distribution) noise to an image."""

    def __init__(self, mean: float = 0.0, std: float = 25.0):
        self._mean = mean
        self._std = std

    @property
    def name(self) -> str:
        return "Gaussian Noise"

    @property
    def description(self) -> str:
        return f"Gaussian noise (μ={self._mean}, σ={self._std})"

    def apply(self, image: np.ndarray) -> np.ndarray:
        noise = np.random.normal(self._mean, self._std, image.shape)
        noisy = image.astype(np.float64) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)


class SaltAndPepperNoise(ImageFilter):
    """Adds salt & pepper (impulse) noise to an image."""

    def __init__(self, density: float = 0.05):
        self._density = density  # Fraction of pixels to corrupt

    @property
    def name(self) -> str:
        return "Salt & Pepper Noise"

    @property
    def description(self) -> str:
        return f"Salt & pepper noise (density={self._density:.2f})"

    def apply(self, image: np.ndarray) -> np.ndarray:
        output = image.copy()
        num_pixels = int(self._density * image.size)

        # Salt (white pixels)
        coords = [np.random.randint(0, dim, num_pixels // 2) for dim in image.shape[:2]]
        if image.ndim == 3:
            output[coords[0], coords[1], :] = 255
        else:
            output[coords[0], coords[1]] = 255

        # Pepper (black pixels)
        coords = [np.random.randint(0, dim, num_pixels // 2) for dim in image.shape[:2]]
        if image.ndim == 3:
            output[coords[0], coords[1], :] = 0
        else:
            output[coords[0], coords[1]] = 0

        return output
