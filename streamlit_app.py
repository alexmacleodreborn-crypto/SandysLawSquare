import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# ==================================================
# App config
# ==================================================
st.set_page_config(
    page_title="Sandy’s Law — Emergent Time Mapping",
    layout="wide"
)

st.title("Sandy’s Law — Emergence of Shared Time")
st.caption("CSV → local time τ → Z–Σ space → event tagging")

# ==================================================
# Utility functions
# ==================================================
def normalize_01(x):
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.full_like(x, 0.5)
    return (x - lo) / (hi - lo)

def rolling_median(y, win):
    if win <= 1:
        return y
    return pd.Series(y).rolling(win, center=True, min_periods=1).median().to_numpy()

def grad(y, x):
    if len(y) < 3:
        return np.zeros_like(y)
    g = np.gradient(y, x)
   