import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==================================================
# App config
# ==================================================
st.set_page_config(
    page_title="Sandy’s Law — Map Light Curve → (Z, Σ)",
    layout="wide"
)

st.title("Sandy’s Law — Map a Light Curve into Toy Space (Z, Σ)")
st.caption("CSV → preprocess → derive Σ(t) + Z(t) → phase space → corner dwell diagnostics (Toy 3 mapping)")

# ==================================================
# Helpers
# ==================================================
def _as_float_series(x):
    return pd.to_numeric(x, errors="coerce").astype(float)

def normalize_01(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < eps:
        return np.full_like(x, 0.5, dtype=float)
    return (x - lo) / (hi - lo)

def rolling_smooth(y: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return y
    s = pd.Series(y)
    return s.rolling(win, center=True, min_periods=max(3, win // 3)).median().to_numpy()

def robust_clip(y: np.ndarray, qlo: float, qhi: float) -> np.ndarray:
    lo = np.nanquantile(y, qlo)
    hi = np.nanquantile(y, qhi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return y
    return np.clip(y, lo, hi)

def compute_derivative(t: np.ndarray, y: np.ndarray) -> np.ndarray