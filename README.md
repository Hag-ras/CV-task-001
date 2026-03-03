# Image Processing Lab — Task 1

A Streamlit app implementing core image processing operations from scratch using NumPy. Clean, modular design following SOLID principles; intended for learning and experimentation.

---

## Quick start

Install dependencies and run the app:

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Project structure

```
.
├── app.py
├── requirements.txt
└── filters/
    ├── __init__.py
    ├── base.py                 # Abstract base classes (ImageFilter, KernelFilter)
    ├── noise.py                # UniformNoise, GaussianNoise, SaltAndPepperNoise
    ├── smoothing.py            # AverageFilter, GaussianFilter, MedianFilter
    ├── edge.py                 # SobelEdge, RobertsEdge, PrewittEdge, CannyEdge
    └── enhancement.py          # Equalization, Normalization, Thresholding, FrequencyFilters, HybridImageCreator
└── utils/
    └── histogram.py            # Histogram + CDF plotting utilities
```

## What this project does

- Implements common image filters and enhancement techniques from first principles.
- Exposes a Streamlit UI for interactively applying filters and visualizing results.
- Designed for clarity and extensibility: add new filters by subclassing the provided abstractions.

## Highlights & implementation notes

- Convolution is implemented manually using reflect padding to avoid border artefacts.
- Color images are processed per-channel for smoothing and noise operations.
- Edge detectors convert color images to grayscale internally when needed.
- Histogram equalization for color uses YCrCb (luminance equalization preserves hue).
- Canny edge detection uses OpenCV's `cv2.Canny` per project specification; other operations are implemented from scratch.

## Filters (summary)

- Smoothing
    - Average Filter — box kernel via `np.ones / (k*k)`, sliding convolution
    - Gaussian Filter — Gaussian kernel computed and normalized
    - Median Filter — sliding-window median (non-linear)
- Noise
    - Uniform, Gaussian, Salt-and-Pepper noise generators
- Edge Detection
    - Sobel, Roberts, Prewitt — gradient-based detectors implemented manually
    - Canny — uses `cv2.Canny` (spec requirement)
- Enhancement
    - Histogram Equalization — CDF-based LUT mapping
    - Normalization — min-max scaling to [0, 255]
    - Otsu Thresholding — implemented from scratch
    - Frequency Filters — `np.fft.fft2` with circular masks (low/high-pass)
    - Hybrid Images — combine low-frequency of one image with high-frequency of another

## Design principles

- Single Responsibility: each filter class does one thing.
- Open/Closed: new filters can be added by subclassing base abstractions.
- Liskov Substitution: filters share a common `apply(image) -> image` interface.
- Interface Segregation: kernel-specific methods are only required for kernel filters.
- Dependency Inversion: `app.py` depends on abstractions, not concrete filter classes.

## Development

- Follow the established class patterns in `filters/base.py` when adding filters.
- Write unit tests that validate both grayscale and RGB behavior for new filters.

## Team

1. Saif Mohammed
2. Seif Samaan
3. Mohamed Ashraf
4. Fady Osama

