"""
Image Processing Lab — Task 1
Streamlit application implementing all task requirements.
All filters are built from scratch (no cv2.filter2D for smoothing/edge).
"""

import sys
import os

# Ensure the app's own directory is first on the path
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

# Remove any other 'filters' module that may have been cached
for _key in list(sys.modules.keys()):
    if _key == "filters" or _key.startswith("filters."):
        del sys.modules[_key]

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import matplotlib.pyplot as plt

from filters import (
    UniformNoise, GaussianNoise, SaltAndPepperNoise,
    AverageFilter, GaussianFilter, MedianFilter,
    SobelEdge, RobertsEdge, PrewittEdge, CannyEdge,
    HistogramEqualizer, ImageNormalizer,
    LowPassFreqFilter, HighPassFreqFilter, HybridImageCreator,
)
from utils.histogram import plot_gray_histogram, plot_rgb_histograms

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Image Processing Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS — dark terminal aesthetic
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg: #0a0a0f;
    --surface: #12121a;
    --card: #1a1a26;
    --border: #2a2a3d;
    --accent: #7c3aed;
    --accent2: #06b6d4;
    --green: #4ade80;
    --orange: #f97316;
    --text: #e2e8f0;
    --muted: #64748b;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
}

.stApp { background: var(--bg) !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

/* Cards */
.lab-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
}

/* Tag badges */
.badge {
    display: inline-block;
    background: var(--accent);
    color: white;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    padding: 2px 10px;
    border-radius: 20px;
    margin-bottom: 8px;
}
.badge-cyan { background: var(--accent2); }
.badge-green { background: #166534; color: var(--green); }
.badge-orange { background: #7c2d12; color: var(--orange); }

/* Title */
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #7c3aed, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin-bottom: 4px;
}

.hero-sub {
    color: var(--muted);
    font-size: 0.9rem;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.05em;
}

/* Image panels */
.img-label {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 6px;
}

/* Sliders, selects */
.stSlider > div { color: var(--text) !important; }
.stSelectbox label { color: var(--text) !important; }
.stRadio label { color: var(--text) !important; }

div[data-testid="stImage"] img {
    border-radius: 8px;
    border: 1px solid var(--border);
}

/* Section headers */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: var(--accent2);
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin: 20px 0 12px 0;
}

/* Dividers */
hr { border-color: var(--border) !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    padding: 8px 20px !important;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def load_image(upload) -> np.ndarray:
    """Load uploaded file as BGR numpy array."""
    file_bytes = np.frombuffer(upload.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def show_image_pair(original: np.ndarray, processed: np.ndarray,
                    left_label="Original", right_label="Result"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="img-label">{left_label}</div>', unsafe_allow_html=True)
        st.image(bgr_to_rgb(original), use_container_width=True)
    with c2:
        st.markdown(f'<div class="img-label">{right_label}</div>', unsafe_allow_html=True)
        st.image(bgr_to_rgb(processed) if processed.ndim == 3 else processed,
                 use_container_width=True)


def show_directional_trio(gx, gy, mag, names=("Gx", "Gy", "Magnitude")):
    c1, c2, c3 = st.columns(3)
    for col, img, label in zip([c1, c2, c3], [gx, gy, mag], names):
        with col:
            st.markdown(f'<div class="img-label">{label}</div>', unsafe_allow_html=True)
            st.image(img, use_container_width=True)


# ──────────────────────────────────────────────
# Sidebar — image upload
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title">🔬 ImgLab</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">// image_processing_v1</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="section-header">Input</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "bmp", "tiff"])

    uploaded_b = None
    st.markdown("---")
    st.markdown('<div class="section-header">Navigation</div>', unsafe_allow_html=True)

    section = st.radio("", [
        "🏠  Overview",
        "🌫️  Noise",
        "🔵  Smoothing Filters",
        "⚡  Edge Detection",
        "📊  Histogram",
        "✨  Enhancement",
        "🌊  Frequency Domain",
        "🔀  Hybrid Images",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div class="hero-sub" style="font-size:10px;color:#3d3d5c">All filters built from scratch<br>No cv2.filter2D for core ops</div>',
                unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Guard — need an image
# ──────────────────────────────────────────────
if uploaded is None:
    # Clear any cached image when no file is uploaded
    st.session_state.pop("original_image", None)
    st.session_state.pop("image", None)
    st.session_state.pop("noisy_image", None)
    st.markdown("""
    <div style="text-align:center; padding: 80px 20px;">
        <div class="hero-title" style="font-size:3rem;">🔬 Image Processing Lab</div>
        <div class="hero-sub" style="font-size:1rem; margin-top:12px;">
            Upload an image in the sidebar to begin
        </div>
        <div style="margin-top:32px; color:#3d3d5c; font-family:'Space Mono',monospace; font-size:12px;">
            Noise • Smoothing • Edge Detection • Histogram • Equalization<br>
            Normalization • Frequency Domain • Hybrid Images
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Load image ONCE and cache in session state ──
_upload_id = uploaded.file_id
if st.session_state.get("_upload_id") != _upload_id:
    # New file uploaded — decode once and store
    _loaded = load_image(uploaded)
    st.session_state["_upload_id"] = _upload_id
    st.session_state["original_image"] = _loaded
    st.session_state["image"] = _loaded.copy()
    st.session_state.pop("noisy_image", None)

original_image = st.session_state["original_image"]  # never modified
image = st.session_state["image"]                     # current working image


# ══════════════════════════════════════════════
# SECTIONS
# ══════════════════════════════════════════════

# ── Overview ──────────────────────────────────
if section == "🏠  Overview":
    st.markdown('<div class="hero-title">Image Overview</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hero-sub">Shape: {image.shape} • dtype: {image.dtype}</div>',
                unsafe_allow_html=True)
    st.markdown("---")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown('<div class="img-label">Uploaded Image (RGB)</div>', unsafe_allow_html=True)
        st.image(bgr_to_rgb(image), use_container_width=True)
    with c2:
        st.markdown('<div class="img-label">Grayscale</div>', unsafe_allow_html=True)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        st.image(gray, use_container_width=True)
        st.markdown(f"""
        <div class="lab-card" style="margin-top:12px">
            <div style="font-family:'Space Mono',monospace;font-size:12px;color:#64748b">STATS</div>
            <div style="font-size:13px;margin-top:8px">
            Min: <b>{image.min()}</b><br>
            Max: <b>{image.max()}</b><br>
            Mean: <b>{image.mean():.1f}</b><br>
            Std: <b>{image.std():.1f}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Noise ─────────────────────────────────────
elif section == "🌫️  Noise":
    st.markdown('<div class="hero-title">Additive Noise</div>', unsafe_allow_html=True)
    st.markdown('<div class="badge">opencv imread</div>', unsafe_allow_html=True)

    noise_type = st.selectbox("Noise Type", ["Uniform", "Gaussian", "Salt & Pepper"])

    if noise_type == "Uniform":
        col1, col2 = st.columns(2)
        low = col1.slider("Low", -100, 0, -50)
        high = col2.slider("High", 0, 100, 50)
        filt = UniformNoise(low, high)

    elif noise_type == "Gaussian":
        col1, col2 = st.columns(2)
        mean = col1.slider("Mean", -50.0, 50.0, 0.0)
        std = col2.slider("Std Dev (σ)", 1.0, 100.0, 25.0)
        filt = GaussianNoise(mean, std)

    else:
        density = st.slider("Noise Density", 0.01, 0.3, 0.05, 0.01)
        filt = SaltAndPepperNoise(density)

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        apply_pressed = st.button("Apply Noise")
    with btn_col2:
        undo_pressed = st.button("↩ Undo Noise")

    if apply_pressed:
        result = filt.apply(image)
        st.session_state["noisy_image"] = result
        st.session_state["image"] = result  # update working image

    if undo_pressed:
        st.session_state["image"] = original_image.copy()
        st.session_state.pop("noisy_image", None)
        st.rerun()

    if "noisy_image" in st.session_state:
        show_image_pair(original_image, st.session_state["noisy_image"],
                        "Original", f"+ {noise_type} Noise")


# ── Smoothing ─────────────────────────────────
elif section == "🔵  Smoothing Filters":
    st.markdown('<div class="hero-title">Low-Pass Filters</div>', unsafe_allow_html=True)
    st.markdown('<div class="badge badge-green">from scratch — numpy convolution</div>', unsafe_allow_html=True)

    filter_type = st.selectbox("Filter", ["Average", "Gaussian", "Median"])
    ksize = st.select_slider("Kernel Size", [3, 5, 7], value=3)

    if filter_type == "Average":
        filt = AverageFilter(ksize)
    elif filter_type == "Gaussian":
        sigma = st.slider("Sigma (σ)", 0.5, 5.0, 1.0, 0.1)
        filt = GaussianFilter(ksize, sigma)
    else:
        filt = MedianFilter(ksize)

    st.info(f"**{filt.name}** — {filt.description}")

    if st.button("Apply Filter"):
        with st.spinner("Convolving..."):
            result = filt.apply(image)
        show_image_pair(image, result, "Original", filt.name)


# ── Edge Detection ────────────────────────────
elif section == "⚡  Edge Detection":
    st.markdown('<div class="hero-title">Edge Detection</div>', unsafe_allow_html=True)

    detector = st.selectbox("Detector", ["Sobel", "Roberts", "Prewitt", "Canny"])

    if detector == "Canny":
        st.markdown('<div class="badge badge-cyan">OpenCV Canny</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        low_t = col1.slider("Low Threshold", 10, 200, 50)
        high_t = col2.slider("High Threshold", 50, 400, 150)
        filt = CannyEdge(low_t, high_t)

        if st.button("Detect Edges"):
            result = filt.apply(image)
            show_image_pair(image, result, "Original", "Canny Edges")

    else:
        st.markdown('<div class="badge badge-green">from scratch — manual gradient</div>',
                    unsafe_allow_html=True)
        filt_map = {"Sobel": SobelEdge, "Roberts": RobertsEdge, "Prewitt": PrewittEdge}
        filt = filt_map[detector]()

        if st.button("Detect Edges"):
            gx, gy, mag = filt.apply_directional(image)
            st.markdown(f"**{filt.name}** — Directional + Combined")
            show_directional_trio(gx, gy, mag,
                                  (f"{detector} — X direction",
                                   f"{detector} — Y direction",
                                   "Combined Magnitude"))


# ── Histogram ────────────────────────────────
elif section == "📊  Histogram":
    st.markdown('<div class="hero-title">Histogram & Distribution</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Grayscale", "RGB Channels"])

    with tab1:
        fig = plot_gray_histogram(image)
        st.pyplot(fig)
        plt.close(fig)

    with tab2:
        fig = plot_rgb_histograms(image)
        st.pyplot(fig)
        plt.close(fig)


# ── Enhancement ──────────────────────────────
elif section == "✨  Enhancement":
    st.markdown('<div class="hero-title">Enhancement</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Equalization", "Normalization", "Color→Gray"])

    with tab1:
        st.markdown('<div class="badge badge-green">CDF mapping — from scratch</div>',
                    unsafe_allow_html=True)
        filt = HistogramEqualizer()
        result = filt.apply(image)
        show_image_pair(image, result, "Original", "Equalized")

        col1, col2 = st.columns(2)
        with col1:
            fig = plot_gray_histogram(image)
            st.pyplot(fig); plt.close(fig)
        with col2:
            fig = plot_gray_histogram(result)
            st.pyplot(fig); plt.close(fig)

    with tab2:
        st.markdown('<div class="badge badge-green">min-max normalization — from scratch</div>',
                    unsafe_allow_html=True)
        filt = ImageNormalizer()
        result = filt.apply(image)
        show_image_pair(image, result, "Original", "Normalized")


    with tab3:
        st.markdown("### Color → Grayscale + RGB Histograms")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="img-label">Color</div>', unsafe_allow_html=True)
            st.image(bgr_to_rgb(image), use_container_width=True)
        with c2:
            st.markdown('<div class="img-label">Grayscale</div>', unsafe_allow_html=True)
            st.image(gray, use_container_width=True)

        fig = plot_rgb_histograms(image)
        st.pyplot(fig); plt.close(fig)


# ── Frequency Domain ──────────────────────────
elif section == "🌊  Frequency Domain":
    st.markdown('<div class="hero-title">Frequency Domain Filters</div>', unsafe_allow_html=True)
    st.markdown('<div class="badge badge-cyan">numpy FFT — from scratch</div>', unsafe_allow_html=True)

    cutoff = st.slider("Cutoff Frequency (radius in pixels)", 5, 100, 30)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Low-Pass (blur/smooth)")
        lp = LowPassFreqFilter(cutoff)
        lp_result = lp.apply(image)
        st.image(lp_result, use_container_width=True)

    with col2:
        st.markdown("#### High-Pass (edges/sharpen)")
        hp = HighPassFreqFilter(cutoff)
        hp_result = hp.apply(image)
        st.image(hp_result, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="img-label">Original</div>', unsafe_allow_html=True)
    st.image(bgr_to_rgb(image), use_container_width=True)


# ── Hybrid Images ─────────────────────────────
elif section == "🔀  Hybrid Images":
    st.markdown('<div class="hero-title">Hybrid Images</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Low-freq from image A + High-freq from image B</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="badge badge-cyan">numpy FFT — from scratch</div>', unsafe_allow_html=True)

    st.markdown("---")
    uploaded_b = st.file_uploader("Upload second image (for high frequencies)",
                                   type=["png", "jpg", "jpeg", "bmp"])

    if uploaded_b is None:
        st.info("Upload a second image to create the hybrid. The first image (sidebar) provides low frequencies.")
    else:
        image_b = load_image(uploaded_b)

        # Resize image_b to match image_a
        h, w = image.shape[:2]
        image_b_resized = cv2.resize(image_b, (w, h))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="img-label">Image A (low freq)</div>', unsafe_allow_html=True)
            st.image(bgr_to_rgb(image), use_container_width=True)
        with col2:
            st.markdown('<div class="img-label">Image B (high freq)</div>', unsafe_allow_html=True)
            st.image(bgr_to_rgb(image_b_resized), use_container_width=True)
        with col3:
            st.markdown('<div class="img-label">Hybrid</div>', unsafe_allow_html=True)
            low_c = st.slider("Low-freq cutoff", 5, 80, 20)
            high_c = st.slider("High-freq cutoff", 5, 80, 20)
            creator = HybridImageCreator(low_c, high_c)
            hybrid = creator.create(image, image_b_resized)
            st.image(hybrid, use_container_width=True)