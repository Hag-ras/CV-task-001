# 🔬 Image Processing Lab — Task 1

A fully-featured Streamlit application implementing all image processing operations **from scratch** using NumPy, following SOLID principles.

---

## 🚀 Setup & Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure

```
image_processing_app/
├── app.py                    # Streamlit UI (entry point)
├── requirements.txt
├── filters/
│   ├── __init__.py
│   ├── base.py               # Abstract base classes (ImageFilter, KernelFilter)
│   ├── noise.py              # UniformNoise, GaussianNoise, SaltAndPepperNoise
│   ├── smoothing.py          # AverageFilter, GaussianFilter, MedianFilter
│   ├── edge.py               # SobelEdge, RobertsEdge, PrewittEdge, CannyEdge
│   └── enhancement.py        # Equalization, Normalization, Thresholding,
│                             #   FrequencyFilters, HybridImageCreator
└── utils/
    ├── __init__.py
    └── histogram.py          # Histogram + CDF plotting
```

---

## 🏗️ SOLID Principles Applied

| Principle | How |
|---|---|
| **S** — Single Responsibility | Each filter class has one job. `UniformNoise` only adds uniform noise; `HistogramEqualizer` only equalizes. |
| **O** — Open/Closed | `GradientEdgeDetector` is open for extension (just subclass + define `kernel_x`/`kernel_y`) but closed for modification. Adding `LaplacianEdge` requires zero changes to existing code. |
| **L** — Liskov Substitution | Any `ImageFilter` subclass can replace another — they all implement `apply(image) → image`. |
| **I** — Interface Segregation | `KernelFilter` adds only `get_kernel()` for kernel-based filters. Non-kernel filters (e.g. `MedianFilter`, `CannyEdge`) extend `ImageFilter` directly without being forced to implement irrelevant methods. |
| **D** — Dependency Inversion | `app.py` depends on the `ImageFilter` abstraction, not concrete classes. Swapping a filter requires only changing one line in the UI. |

---

## 🔧 Filters Built From Scratch (no cv2.filter2D)

### Smoothing
- **Average Filter** — box kernel via `np.ones / (k*k)`, manual sliding convolution
- **Gaussian Filter** — kernel from `exp(-(x²+y²)/2σ²)`, normalized
- **Median Filter** — sliding window, `np.median` per patch (non-linear, no convolution)

### Edge Detection
- **Sobel** — 3×3 Gx/Gy kernels, `magnitude = sqrt(Gx²+Gy²)`
- **Roberts** — 2×2 cross-difference kernels
- **Prewitt** — 3×3 uniform gradient kernels
- **Canny** — OpenCV `cv2.Canny` (as required by task spec)

### Enhancement
- **Histogram Equalization** — CDF-based LUT mapping, from scratch
- **Normalization** — min-max stretching `(x - min) / (max - min) * 255`
- **Otsu Thresholding** — between-class variance maximization, from scratch
- **Frequency Filters** — `np.fft.fft2` + circular mask (low-pass/high-pass)
- **Hybrid Images** — blend low-freq(A) + high-freq(B) in frequency domain

---

## 📌 Notes
- Convolution uses **reflect padding** to avoid black borders
- Color images are handled channel-by-channel for smoothing/noise
- Edge detectors auto-convert to grayscale internally
- Histogram equalization uses YCrCb space for color images (preserves hue)
# CV-task-01
# CV-task-001
