import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# ==================================================
# App config
# ==================================================
st.set_page_config(
    page_title="Sandy’s Law — Square (Toy 3)",
    layout="wide"
)

st.title("Sandy’s Law — CSV → Square → Corner Dwell")
st.caption("Paste light-curve data → Toy 3 diagnostic")

# ==================================================
# Helper functions
# ==================================================
def normalize_01(x):
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi - lo < 1e-12:
        return np.full_like(x, 0.5)
    return (x - lo) / (hi - lo)

def rolling_median(y, win):
    if win <= 1:
        return y
    return pd.Series(y).rolling(win, center=True, min_periods=1).median().to_numpy()

def derivative(t, y):
    if len(t) < 3:
        return np.zeros_like(y)
    return np.gradient(y, t)

def corner_mask(Z, S, th):
    hi = th
    lo = 1 - th
    return (
        ((Z >= hi) & (S >= hi)) |
        ((Z <= lo) & (S >= hi)) |
        ((Z >= hi) & (S <= lo)) |
        ((Z <= lo) & (S <= lo))
    )

def plot_phase(Z, S, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(Z, S, lw=1.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Z (trap strength)")
    ax.set_ylabel("Σ (entropy escape)")
    ax.set_title(title)
    ax.grid(alpha=0.35)
    st.pyplot(fig)
    plt.close(fig)

def plot_timeseries(t, Z, S):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t, Z, label="Z")
    ax.plot(t, S, label="Σ")
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    ax.legend()
    ax.grid(alpha=0.35)
    st.pyplot(fig)
    plt.close(fig)

def plot_lc(t, f, fs):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t, f, alpha=0.5, label="raw")
    ax.plot(t, fs, lw=1.2, label="smoothed")
    ax.set_xlabel("time")
    ax.set_ylabel("flux")
    ax.legend()
    ax.grid(alpha=0.35)
    st.pyplot(fig)
    plt.close(fig)

# ==================================================
# CSV PASTE INPUT
# ==================================================
st.subheader("Paste CSV Data")

csv_text = st.text_area(
    "Paste CSV here (must contain time and flux columns)",
    height=220,
    placeholder="time,flux\n0.0,1.01\n0.1,1.02\n0.2,1.03"
)

if not csv_text.strip():
    st.info("Paste CSV data above to begin.")
    st.stop()

try:
    df = pd.read_csv(StringIO(csv_text))
except Exception as e:
    st.error(f"CSV parsing failed: {e}")
    st.stop()

if df.shape[1] < 2:
    st.error("CSV must contain at least two columns.")
    st.stop()

time_col = st.selectbox("Time column", df.columns, index=0)
flux_col = st.selectbox("Flux column", df.columns, index=1)

t = df[time_col].astype(float).to_numpy()
f = df[flux_col].astype(float).to_numpy()

mask = np.isfinite(t) & np.isfinite(f)
t, f = t[mask], f[mask]

order = np.argsort(t)
t, f = t[order], f[order]

# ==================================================
# Controls
# ==================================================
st.sidebar.header("Processing")

smooth = st.sidebar.slider("Smoothing window", 1, 51, 11, 2)
early_frac = st.sidebar.slider("Early fraction", 0.05, 1.0, 0.25, 0.05)
corner_th = st.sidebar.slider("Corner threshold", 0.6, 0.95, 0.85, 0.01)

z_mode = st.sidebar.radio(
    "Z definition",
    [
        "Z = 1 − |dΣ/dt| (trap = stability)",
        "Z = |dΣ/dt| (trap = rapid change)"
    ]
)

# ==================================================
# Build Σ and Z
# ==================================================
f_s = rolling_median(f, smooth)
Sigma = normalize_01(f_s)

dS = np.abs(derivative(t, Sigma))
dS_n = normalize_01(dS)

if z_mode.startswith("Z = 1"):
    Z = 1 - dS_n
else:
    Z = dS_n

Z = np.clip(Z, 0, 1)
Sigma = np.clip(Sigma, 0, 1)

n_early = max(10, int(len(t) * early_frac))
t_e = t[:n_early]
Z_e = Z[:n_early]
S_e = Sigma[:n_early]

# ==================================================
# Plots
# ==================================================
st.subheader("Light Curve")
plot_lc(t, f, f_s)

st.subheader("Time Series")
plot_timeseries(t, Z, Sigma)

c1, c2 = st.columns(2)
with c1:
    plot_phase(Z, Sigma, "Phase Space (Full)")
with c2:
    plot_phase(Z_e, S_e, "Phase Space (Early)")

# ==================================================
# Toy 3 Diagnostic
# ==================================================
st.subheader("🔴 Toy 3 — Corner Dwell Diagnostic")

corner_hits = corner_mask(Z_e, S_e, corner_th)
dwell_frac = corner_hits.mean()

st.metric("Corner dwell fraction", f"{100*dwell_frac:.2f}%")

if dwell_frac < 0.03:
    st.success("Stable early evolution (no trapping)")
elif dwell_frac < 0.10:
    st.warning("Precursor structure detected")
else:
    st.error("Strong corner locking → imminent transition")

# ==================================================
# Interpretation
# ==================================================
with st.expander("Physical interpretation (Sandy’s Law)"):
    st.markdown(
        """
**Σ (entropy escape)** is the observable photon output.  
**Z (trap strength)** measures how constrained that output is.

**Toy 3 result:**  
Systems approaching a transition **spend measurable time trapped in corners**
of (Z, Σ) phase space.

This dwell is **observable in early light curves**.
"""
    )