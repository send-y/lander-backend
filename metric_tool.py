# metric_tool.py
from __future__ import annotations
__all__ = ["ImageMetrics", "FEATURE_COLS", "compute_metrics", "load_image_rgb"]

import math
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from PIL.ExifTags import TAGS as _EXIF_TAGS

try:
    import cv2  # type: ignore
    _HAVE_CV2 = True
except Exception:
    cv2 = None
    _HAVE_CV2 = False

try:
    from scipy.ndimage import convolve as _scipy_convolve  # type: ignore
    from scipy.ndimage import gaussian_filter as _scipy_gaussian  # type: ignore
    from scipy.fft import dctn as _scipy_dctn  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# =============================================================================
# Feature columns
# =============================================================================

FEATURE_COLS: list[str] = [
    # basic stats
    "entropy",
    "brightness_mean",
    "brightness_std",
    "skewness_brightness",
    "kurtosis_brightness",

    # sharpness/edges/gradients
    "laplacian_variance",
    "gradient_mean",
    "gradient_variance",
    "edge_density",

    # frequency-domain
    "hf_energy_ratio",
    "spectral_slope",
    "spectral_flatness",
    "fft_phase_entropy",
    "freq_lf",
    "freq_mf",
    "freq_hf",

    # transform-domain artifacts
    "dct_energy_ratio",
    "jpeg_artifact_score",
    "stripe_score",

    # texture
    "lbp_entropy",
    "glcm_contrast",
    "glcm_homogeneity",
    "glcm_energy",

    # noise
    "noise_entropy",
    "residual_std",
    "noise_variance_patch_std",

    # color
    "saturation_mean",
    "hue_entropy",
    "rgb_channel_corr",
    "colorfulness",
    "gray_world_error",

    # compressibility
    "compression_ratio",

    # === NEW METRICS ===
    "grid_regularity",
    "color_uniformity",
    "symmetry_score",
    "fractal_dimension",
    "noise_naturalness",
]


# =============================================================================
# EXIF helpers
# =============================================================================

_TAG_NAME_TO_ID: dict[str, int] = {name: tag_id for tag_id, name in _EXIF_TAGS.items()}
_EXIF_SOFTWARE = _TAG_NAME_TO_ID.get("Software", 305)
_EXIF_MAKE = _TAG_NAME_TO_ID.get("Make", 271)
_EXIF_MODEL = _TAG_NAME_TO_ID.get("Model", 272)

_AI_SOFTWARE_KEYWORDS: frozenset[str] = frozenset({
    "stable diffusion", "midjourney", "dall-e", "comfyui",
    "automatic1111", "sdxl", "novelai", "adobe firefly", "imagen",
})


# =============================================================================
# Data structure
# =============================================================================

@dataclass(frozen=True)
class ImageMetrics:
    # basic stats
    entropy: float
    brightness_mean: float
    brightness_std: float
    skewness_brightness: float
    kurtosis_brightness: float

    # sharpness
    laplacian_variance: float
    gradient_mean: float
    gradient_variance: float
    edge_density: float

    # frequency
    hf_energy_ratio: float
    spectral_slope: float
    spectral_flatness: float
    fft_phase_entropy: float
    freq_lf: float
    freq_mf: float
    freq_hf: float

    # artifacts
    dct_energy_ratio: float
    jpeg_artifact_score: float
    stripe_score: float

    # texture
    lbp_entropy: float
    glcm_contrast: float
    glcm_homogeneity: float
    glcm_energy: float

    # noise
    noise_entropy: float
    residual_std: float
    noise_variance_patch_std: float

    # color
    saturation_mean: float
    hue_entropy: float
    rgb_channel_corr: float
    colorfulness: float
    gray_world_error: float

    # compressibility
    compression_ratio: float

    # === NEW ===
    grid_regularity: float
    color_uniformity: float
    symmetry_score: float
    fractal_dimension: float
    noise_naturalness: float

    # extras (debug only)
    metadata_flag: int
    exif_present: int
    orig_width: int
    orig_height: int
    orig_aspect: float
    file_size_kb: float
    file_bpp: float


# =============================================================================
# Loading + preprocessing
# =============================================================================

def load_image_rgb(path: str | Path) -> Image.Image:
    p = Path(path)
    with Image.open(p) as img:
        img.load()
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img.copy()


def _pil_to_rgb_u8(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB image; got shape={arr.shape}")
    return arr


def _rgb_to_gray01(rgb_u8: np.ndarray) -> np.ndarray:
    r = rgb_u8[..., 0].astype(np.float64)
    g = rgb_u8[..., 1].astype(np.float64)
    b = rgb_u8[..., 2].astype(np.float64)
    gray = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    return np.clip(gray, 0.0, 1.0)


def _center_crop_to_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def _prepare_for_analysis(pil_img: Image.Image, analysis_size: int = 512) -> Image.Image:
    img = _center_crop_to_square(pil_img)
    if img.size != (analysis_size, analysis_size):
        img = img.resize((analysis_size, analysis_size), resample=Image.Resampling.LANCZOS)
    return img


# =============================================================================
# Math helpers
# =============================================================================

def _gray_to_u8(gray01: np.ndarray) -> np.ndarray:
    return np.clip(gray01 * 255.0, 0, 255).astype(np.uint8)


def _normalize01(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    lo = float(x.min())
    hi = float(x.max())
    if hi - lo < eps:
        return np.zeros_like(x, dtype=np.float64)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _shannon_entropy_u8(u8: np.ndarray, bins: int = 256) -> float:
    hist = np.bincount(u8.ravel(), minlength=bins).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.0
    p = hist / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _convolve2d(gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    if _HAVE_SCIPY:
        return _scipy_convolve(gray.astype(np.float64), kernel, mode="mirror")
    from numpy.lib.stride_tricks import sliding_window_view
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(gray.astype(np.float64), ((ph, ph), (pw, pw)), mode="reflect")
    windows = sliding_window_view(padded, (kh, kw))
    return np.einsum("hwij,ij->hw", windows, kernel[::-1, ::-1])


def _gaussian_blur(gray: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    if _HAVE_CV2:
        k = int(max(3, math.ceil(sigma * 6))) | 1
        return cv2.GaussianBlur(gray.astype(np.float64), (k, k), sigmaX=sigma, sigmaY=sigma)
    if _HAVE_SCIPY:
        return _scipy_gaussian(gray.astype(np.float64), sigma=sigma)
    k = max(3, int(round(sigma * 6 + 1))) | 1
    kernel = np.ones((k, k), dtype=np.float64) / (k * k)
    return _convolve2d(gray, kernel)


def _sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    if _HAVE_CV2:
        u8 = _gray_to_u8(gray)
        gx = cv2.Sobel(u8, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(u8, cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(gx * gx + gy * gy)
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    gx = _convolve2d(gray, kx)
    gy = _convolve2d(gray, ky)
    return np.sqrt(gx * gx + gy * gy)


def _skew_kurt(gray: np.ndarray) -> tuple[float, float]:
    x = gray.ravel().astype(np.float64)
    mu = x.mean()
    std = x.std()
    if std < 1e-12:
        return 0.0, 0.0
    z = (x - mu) / std
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4) - 3.0)
    return skew, kurt


# =============================================================================
# Original metrics
# =============================================================================

def compute_entropy(gray01: np.ndarray) -> float:
    return _shannon_entropy_u8(_gray_to_u8(gray01))


def compute_brightness_stats(gray01: np.ndarray) -> tuple[float, float, float, float]:
    mean = float(gray01.mean())
    std = float(gray01.std())
    skew, kurt = _skew_kurt(gray01)
    return mean, std, skew, kurt


def compute_laplacian_variance(gray01: np.ndarray) -> float:
    if _HAVE_CV2:
        lap = cv2.Laplacian(_gray_to_u8(gray01), cv2.CV_64F)
        return float(lap.var())
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    lap = _convolve2d(gray01, k)
    return float((lap * 255.0).var())


def compute_gradient_stats(gray01: np.ndarray) -> tuple[float, float]:
    mag = _sobel_magnitude(gray01)
    return float(mag.mean()), float(mag.var())


def compute_edge_density(gray01: np.ndarray, canny_low: int = 100, canny_high: int = 200) -> float:
    h, w = gray01.shape
    n = h * w
    if n == 0:
        return 0.0
    if _HAVE_CV2:
        edges = cv2.Canny(_gray_to_u8(gray01), threshold1=canny_low, threshold2=canny_high)
        return float((edges > 0).sum()) / n
    mag = _sobel_magnitude(gray01)
    thr = mag.mean() + mag.std()
    return float((mag > thr).sum()) / n


def compute_hf_energy_ratio(gray01: np.ndarray, lf_radius_frac: float = 0.25) -> float:
    F = np.fft.fftshift(np.fft.fft2(gray01))
    power = np.abs(F) ** 2
    h, w = gray01.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r0 = lf_radius_frac * (min(h, w) / 2.0)
    total = float(power.sum())
    if total <= 0:
        return 0.0
    return float(power[r > r0].sum()) / total


def compute_frequency_bands(gray01: np.ndarray) -> tuple[float, float, float]:
    F = np.fft.fftshift(np.fft.fft2(gray01))
    power = np.abs(F) ** 2
    total = float(power.sum())
    if total <= 0:
        return 0.0, 0.0, 0.0
    h, w = gray01.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = min(h, w) / 2.0
    lf = float(power[r <= r_max * 0.10].sum()) / total
    mf = float(power[(r > r_max * 0.10) & (r <= r_max * 0.40)].sum()) / total
    hf = float(power[r > r_max * 0.40].sum()) / total
    return lf, mf, hf


def compute_spectral_slope(gray01: np.ndarray, eps: float = 1e-12) -> float:
    F = np.fft.fftshift(np.fft.fft2(gray01))
    power = (np.abs(F) ** 2).astype(np.float64)
    h, w = gray01.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_int = r.astype(np.int32)
    r_max = int(r_int.max())
    sums = np.bincount(r_int.ravel(), weights=power.ravel(), minlength=r_max + 1)
    cnts = np.bincount(r_int.ravel(), minlength=r_max + 1)
    radial = sums / np.maximum(cnts, 1).astype(np.float64)
    freqs = np.arange(1, r_max + 1, dtype=np.float64)
    vals = radial[1:r_max + 1].astype(np.float64)
    mask = vals > 0
    freqs, vals = freqs[mask], vals[mask]
    if len(freqs) < 10:
        return 0.0
    b, _ = np.polyfit(np.log(freqs + eps), np.log(vals + eps), 1)
    return float(-b)


def compute_spectral_flatness(gray01: np.ndarray, eps: float = 1e-12) -> float:
    F = np.fft.fft2(gray01)
    power = (np.abs(F).ravel() ** 2) + eps
    geo = np.exp(np.mean(np.log(power)))
    ar = np.mean(power)
    return float(geo / (ar + eps))


def compute_fft_phase_entropy(gray01: np.ndarray) -> float:
    F = np.fft.fft2(gray01)
    phase = np.angle(F)
    ph01 = (phase + np.pi) / (2 * np.pi)
    u8 = np.clip(ph01 * 255.0, 0, 255).astype(np.uint8)
    return _shannon_entropy_u8(u8)


def compute_dct_energy_ratio(gray01: np.ndarray, patch_size: int = 128) -> float:
    h, w = gray01.shape
    cy, cx = h // 2, w // 2
    half = patch_size // 2
    y0, y1 = max(0, cy - half), min(h, cy + half)
    x0, x1 = max(0, cx - half), min(w, cx + half)
    patch = gray01[y0:y1, x0:x1]
    if patch.shape[0] < 16 or patch.shape[1] < 16:
        patch = gray01[:min(h, patch_size), :min(w, patch_size)]
    if _HAVE_SCIPY:
        dct = _scipy_dctn(patch.astype(np.float64), norm="ortho")
    else:
        dct = np.fft.fft2(patch.astype(np.float64)).real
    power = dct ** 2
    total = float(power.sum())
    if total <= 1e-12:
        return 0.0
    ph, pw = patch.shape
    hf = float(power[ph // 2:, pw // 2:].sum())
    return hf / total


def compute_jpeg_artifact_score(gray01: np.ndarray) -> float:
    u8 = _gray_to_u8(gray01)
    h, w = u8.shape
    h_diffs = []
    v_diffs = []
    for y in range(8, h, 8):
        diff = np.abs(u8[y, :].astype(np.float64) - u8[y - 1, :].astype(np.float64)).mean()
        h_diffs.append(diff)
    for x in range(8, w, 8):
        diff = np.abs(u8[:, x].astype(np.float64) - u8[:, x - 1].astype(np.float64)).mean()
        v_diffs.append(diff)
    all_h = np.abs(np.diff(u8.astype(np.float64), axis=0)).mean()
    all_v = np.abs(np.diff(u8.astype(np.float64), axis=1)).mean()
    block_h = float(np.mean(h_diffs)) if h_diffs else 0.0
    block_v = float(np.mean(v_diffs)) if v_diffs else 0.0
    score_h = block_h / (all_h + 1e-9)
    score_v = block_v / (all_v + 1e-9)
    return float((score_h + score_v) / 2.0)


def compute_stripe_score(gray01: np.ndarray) -> float:
    g = gray01.astype(np.float64)
    row = g.mean(axis=1)
    col = g.mean(axis=0)
    row = row - row.mean()
    col = col - col.mean()
    return float(row.var() + col.var())


def compute_lbp_entropy(gray01: np.ndarray) -> float:
    u8 = _gray_to_u8(gray01)
    neighbors = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
    lbp = np.zeros_like(u8, dtype=np.uint8)
    for dy, dx in neighbors:
        shifted = np.roll(np.roll(u8, dy, axis=0), dx, axis=1)
        lbp = (lbp << 1) | (u8 >= shifted).astype(np.uint8)
    hist = np.bincount(lbp.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.0
    p = hist / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def compute_glcm_features(gray01: np.ndarray, levels: int = 16) -> tuple[float, float, float]:
    g = gray01
    if g.shape[0] > 256:
        img = Image.fromarray(_gray_to_u8(g))
        img = img.resize((256, 256), resample=Image.Resampling.BILINEAR)
        g = np.asarray(img, dtype=np.uint8) / 255.0
    q = np.floor(g * (levels - 1)).astype(np.int32)
    a = q[:, :-1].ravel()
    b = q[:, 1:].ravel()
    glcm = np.zeros((levels, levels), dtype=np.float64)
    np.add.at(glcm, (a, b), 1.0)
    s = glcm.sum()
    if s <= 0:
        return 0.0, 0.0, 0.0
    p = glcm / s
    i = np.arange(levels)[:, None]
    j = np.arange(levels)[None, :]
    contrast = float(((i - j) ** 2 * p).sum())
    homogeneity = float((p / (1.0 + np.abs(i - j))).sum())
    energy = float((p ** 2).sum())
    return contrast, homogeneity, energy


def compute_noise_metrics(gray01: np.ndarray, sigma: float = 1.0) -> tuple[float, float]:
    blurred = _gaussian_blur(gray01, sigma=sigma)
    residual = gray01 - blurred
    residual01 = _normalize01(residual)
    noise_entropy = _shannon_entropy_u8(_gray_to_u8(residual01))
    residual_std = float(residual.std())
    return noise_entropy, residual_std


def compute_noise_variance_patch_std(gray01: np.ndarray, patch: int = 32, sigma: float = 1.0) -> float:
    blurred = _gaussian_blur(gray01, sigma=sigma)
    residual = gray01 - blurred
    h, w = residual.shape
    vars_ = []
    for y in range(0, h - patch + 1, patch):
        for x in range(0, w - patch + 1, patch):
            vars_.append(float(residual[y:y+patch, x:x+patch].var()))
    if len(vars_) < 2:
        return 0.0
    return float(np.std(vars_))


def compute_saturation_mean(rgb_u8: np.ndarray) -> float:
    if _HAVE_CV2:
        hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
        return float(hsv[:, :, 1].mean()) / 255.0
    r = rgb_u8[..., 0].astype(np.float64) / 255.0
    g = rgb_u8[..., 1].astype(np.float64) / 255.0
    b = rgb_u8[..., 2].astype(np.float64) / 255.0
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    diff = cmax - cmin
    sat = np.where(cmax > 1e-9, diff / cmax, 0.0)
    return float(sat.mean())


def compute_hue_entropy(rgb_u8: np.ndarray) -> float:
    if _HAVE_CV2:
        hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
        h = hsv[:, :, 0].astype(np.uint8)
        h255 = np.clip(h.astype(np.float64) * (255.0 / 179.0), 0, 255).astype(np.uint8)
        return _shannon_entropy_u8(h255)
    r = rgb_u8[..., 0].astype(np.float64) / 255.0
    g = rgb_u8[..., 1].astype(np.float64) / 255.0
    b = rgb_u8[..., 2].astype(np.float64) / 255.0
    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g) ** 2 + (r - b) * (g - b)) + 1e-9
    theta = np.arccos(np.clip(num / den, -1.0, 1.0))
    hue = np.where(b <= g, theta, 2 * np.pi - theta)
    h01 = hue / (2 * np.pi)
    u8 = np.clip(h01 * 255.0, 0, 255).astype(np.uint8)
    return _shannon_entropy_u8(u8)


def compute_rgb_channel_corr(rgb_u8: np.ndarray) -> float:
    r = rgb_u8[..., 0].astype(np.float64).ravel()
    g = rgb_u8[..., 1].astype(np.float64).ravel()
    b = rgb_u8[..., 2].astype(np.float64).ravel()

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        sa = a.std()
        sb = b.std()
        if sa < 1e-9 or sb < 1e-9:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    rg = corr(r, g)
    rb = corr(r, b)
    gb = corr(g, b)
    return float((abs(rg) + abs(rb) + abs(gb)) / 3.0)


def compute_colorfulness(rgb_u8: np.ndarray) -> float:
    r = rgb_u8[..., 0].astype(np.float64)
    g = rgb_u8[..., 1].astype(np.float64)
    b = rgb_u8[..., 2].astype(np.float64)
    rg = r - g
    yb = 0.5 * (r + g) - b
    std_rg = rg.std()
    std_yb = yb.std()
    mean_rg = rg.mean()
    mean_yb = yb.mean()
    return float(np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2))


def compute_gray_world_error(rgb_u8: np.ndarray) -> float:
    m = rgb_u8.reshape(-1, 3).astype(np.float64).mean(axis=0)
    r, g, b = float(m[0]), float(m[1]), float(m[2])
    return float(abs(r - g) + abs(g - b)) / 255.0


def compute_compression_ratio(gray01: np.ndarray) -> float:
    raw = _gray_to_u8(gray01).tobytes()
    if not raw:
        return 1.0
    comp = zlib.compress(raw, level=6)
    return float(len(comp)) / float(len(raw))


# =============================================================================
# NEW metrics
# =============================================================================

def compute_grid_regularity(gray01: np.ndarray) -> float:
    F = np.fft.fft2(gray01)
    power = np.abs(F) ** 2
    peaks = np.sort(power.ravel())[-100:]
    return float(peaks.std() / (peaks.mean() + 1e-9))


def compute_color_uniformity(rgb_u8: np.ndarray) -> float:
    r_std = float(rgb_u8[..., 0].astype(np.float64).std())
    g_std = float(rgb_u8[..., 1].astype(np.float64).std())
    b_std = float(rgb_u8[..., 2].astype(np.float64).std())
    return float((r_std + g_std + b_std) / 3.0)


def compute_symmetry_score(gray01: np.ndarray) -> float:
    flipped = np.fliplr(gray01)
    diff = np.abs(gray01 - flipped)
    return float(1.0 - diff.mean())


def compute_fractal_dimension(gray01: np.ndarray) -> float:
    u8 = _gray_to_u8(gray01)
    sizes = [2, 4, 8, 16, 32, 64]
    counts = []
    for size in sizes:
        h, w = u8.shape
        h2 = (h // size) * size
        w2 = (w // size) * size
        blocks = u8[:h2, :w2].reshape(h // size, size, w // size, size)
        count = int((blocks.max(axis=(1, 3)) > 128).sum())
        counts.append(max(count, 1))
    coeffs = np.polyfit(np.log(sizes), np.log(np.array(counts, dtype=np.float64)), 1)
    return float(-coeffs[0])


def compute_noise_naturalness(gray01: np.ndarray) -> float:
    blurred = _gaussian_blur(gray01, sigma=0.5)
    residual = gray01 - blurred
    flat = residual.ravel().astype(np.float64)
    mu = flat.mean()
    std = flat.std() + 1e-9
    z = (flat - mu) / std
    kurt = float(np.mean(z ** 4) - 3.0)
    return float(abs(kurt))


# =============================================================================
# Metadata helpers
# =============================================================================

def compute_exif_flags(pil_img: Image.Image) -> tuple[int, int]:
    try:
        exif = pil_img.getexif()
    except Exception:
        return 0, 1
    if not exif:
        return 0, 1
    
    software = str(exif.get(_EXIF_SOFTWARE, "")).lower().strip()
    make = str(exif.get(_EXIF_MAKE, "")).strip()
    model = str(exif.get(_EXIF_MODEL, "")).strip()
    
    if any(kw in software for kw in _AI_SOFTWARE_KEYWORDS):
        return 1, 1
    if make or model:
        return 1, 0
    return 1, 1

def compute_file_info(path: Path, width: int, height: int) -> tuple[float, float]:
    try:
        size_bytes = path.stat().st_size
    except Exception:
        size_bytes = 0
    px = max(1, width * height)
    return float(size_bytes) / 1024.0, float(size_bytes) / float(px)


# =============================================================================
# Orchestration
# =============================================================================

def compute_metrics(path: str | Path, analysis_size: int = 512) -> ImageMetrics:
    p = Path(path)
    pil_orig = load_image_rgb(p)

    ow, oh = pil_orig.size
    orig_aspect = float(ow) / float(max(1, oh))

    exif_present, metadata_flag = compute_exif_flags(pil_orig)
    file_size_kb, file_bpp = compute_file_info(p, ow, oh)

    pil_img = _prepare_for_analysis(pil_orig, analysis_size=analysis_size)
    rgb = _pil_to_rgb_u8(pil_img)
    gray = _rgb_to_gray01(rgb)

    # basic stats
    entropy = compute_entropy(gray)
    b_mean, b_std, b_skew, b_kurt = compute_brightness_stats(gray)

    # sharpness
    lap_var = compute_laplacian_variance(gray)
    grad_mean, grad_var = compute_gradient_stats(gray)
    edge_den = compute_edge_density(gray)

    # frequency
    hf_ratio = compute_hf_energy_ratio(gray)
    slope = compute_spectral_slope(gray)
    flat = compute_spectral_flatness(gray)
    phase_ent = compute_fft_phase_entropy(gray)
    lf, mf, hf = compute_frequency_bands(gray)

    # artifacts
    dct_ratio = compute_dct_energy_ratio(gray)
    jpeg_score = compute_jpeg_artifact_score(gray)
    stripe_score = compute_stripe_score(gray)

    # texture
    lbp_ent = compute_lbp_entropy(gray)
    glcm_contrast, glcm_homog, glcm_energy = compute_glcm_features(gray, levels=16)

    # noise
    noise_ent, residual_std = compute_noise_metrics(gray)
    noise_patch_std = compute_noise_variance_patch_std(gray)

    # color
    sat = compute_saturation_mean(rgb)
    hue_ent = compute_hue_entropy(rgb)
    rgb_corr = compute_rgb_channel_corr(rgb)
    colorful = compute_colorfulness(rgb)
    gw_err = compute_gray_world_error(rgb)

    # compressibility
    comp_ratio = compute_compression_ratio(gray)

    # === NEW ===
    grid_reg = compute_grid_regularity(gray)
    color_uni = compute_color_uniformity(rgb)
    symmetry = compute_symmetry_score(gray)
    fractal = compute_fractal_dimension(gray)
    noise_nat = compute_noise_naturalness(gray)

    return ImageMetrics(
        entropy=entropy,
        brightness_mean=b_mean,
        brightness_std=b_std,
        skewness_brightness=b_skew,
        kurtosis_brightness=b_kurt,

        laplacian_variance=lap_var,
        gradient_mean=grad_mean,
        gradient_variance=grad_var,
        edge_density=edge_den,

        hf_energy_ratio=hf_ratio,
        spectral_slope=slope,
        spectral_flatness=flat,
        fft_phase_entropy=phase_ent,
        freq_lf=lf,
        freq_mf=mf,
        freq_hf=hf,

        dct_energy_ratio=dct_ratio,
        jpeg_artifact_score=jpeg_score,
        stripe_score=stripe_score,

        lbp_entropy=lbp_ent,
        glcm_contrast=glcm_contrast,
        glcm_homogeneity=glcm_homog,
        glcm_energy=glcm_energy,

        noise_entropy=noise_ent,
        residual_std=residual_std,
        noise_variance_patch_std=noise_patch_std,

        saturation_mean=sat,
        hue_entropy=hue_ent,
        rgb_channel_corr=rgb_corr,
        colorfulness=colorful,
        gray_world_error=gw_err,

        compression_ratio=comp_ratio,

        # === NEW ===
        grid_regularity=grid_reg,
        color_uniformity=color_uni,
        symmetry_score=symmetry,
        fractal_dimension=fractal,
        noise_naturalness=noise_nat,

        # extras
        metadata_flag=metadata_flag,
        exif_present=exif_present,
        orig_width=int(ow),
        orig_height=int(oh),
        orig_aspect=float(orig_aspect),
        file_size_kb=float(file_size_kb),
        file_bpp=float(file_bpp),
    )


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    import argparse, json

    ap = argparse.ArgumentParser(description="Compute strong AI/Real metrics.")
    ap.add_argument("image", help="Path to image")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--size", type=int, default=512, help="analysis_size (default 512)")
    args = ap.parse_args()

    m = compute_metrics(args.image, analysis_size=args.size)

    if args.json:
        print(json.dumps(m.__dict__, ensure_ascii=False, indent=2))
    else:
        for k, v in m.__dict__.items():
            if isinstance(v, float):
                print(f"{k:>24} = {v:.6f}")
            else:
                print(f"{k:>24} = {v}")


if __name__ == "__main__":
    main()