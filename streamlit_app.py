import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# ==================================================
# App config
# ==================================================
st.set_page_config(page_title="Sandy’s Law — Square (Local Time τ)", layout="wide")
st.title("Sandy’s Law — Square Mapping with Local Time (τ)")
st.caption("Paste CSV → build Σ → build local time τ (NOT constant) → compute Z, corner dwell, quench")

# ==================================================
# Helpers
# ==================================================
def normalize_01(x):
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-12:
        return np.full_like(x, 0.5, dtype=float)
    return (x - lo) / (hi - lo)

def rolling_median(y, win):
    if win <= 1:
        return y
    return pd.Series(y).rolling(win, center=True, min_periods=1).median().to_numpy()

def safe_gradient(y, x):
    # gradient dy/dx; robust to uneven x
    if len(y) < 3:
        return np.zeros_like(y, dtype=float)
    g = np.gradient(y, x)
    g[~np.isfinite(g)] = 0.0
    return g

def corner_mask(Z, S, th, which):
    hi = th
    lo = 1.0 - th
    if which == "UR":  # Z high, Σ high
        return (Z >= hi) & (S >= hi)
    if which == "UL":  # Z low, Σ high
        return (Z <= lo) & (S >= hi)
    if which == "LR":  # Z high, Σ low
        return (Z >= hi) & (S <= lo)
    if which == "LL":  # Z low, Σ low
        return (Z <= lo) & (S <= lo)
    return np.zeros_like(Z, dtype=bool)

def plot_phase(Z, S, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(Z, S, lw=1.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Z (Trap Strength)")
    ax.set_ylabel("Σ (Escape)")
    ax.set_title(title)
    ax.grid(alpha=0.35)
    st.pyplot(fig)
    plt.close(fig)

def plot_series(x, series_dict, title, xlabel):
    fig, ax = plt.subplots(figsize=(8, 3))
    for k, v in series_dict.items():
        ax.plot(x, v, label=k)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("value")
    ax.legend()
    ax.grid(alpha=0.35)
    st.pyplot(fig)
    plt.close(fig)

# ==================================================
# Paste CSV input
# ==================================================
st.subheader("Paste CSV")
csv_text = st.text_area(
    "Paste CSV here (needs at least two columns: time, flux OR any two columns you pick below).",
    height=220,
    placeholder="time,flux\n0.0,1.01\n0.1,1.02\n0.2,1.03\n..."
)

if not csv_text.strip():
    st.info("Paste CSV data to begin.")
    st.stop()

try:
    df = pd.read_csv(StringIO(csv_text))
except Exception as e:
    st.error(f"CSV parsing failed: {e}")
    st.stop()

if df.shape[1] < 2:
    st.error("CSV must contain at least two columns.")
    st.stop()

with st.expander("Preview"):
    st.dataframe(df.head(30), use_container_width=True)

cols = list(df.columns)
time_col = st.selectbox("Select time column (can be any monotonic coordinate)", cols, index=0)
flux_col = st.selectbox("Select flux column", cols, index=1)

# Controls
st.sidebar.header("Preprocess")
smooth_win = st.sidebar.slider("Smoothing window (median)", 1, 51, 11, 2)
use_mag = st.sidebar.checkbox("Flux column is magnitude", value=False)

st.sidebar.header("Local time (τ) — choose your clock")
tau_mode = st.sidebar.radio(
    "τ definition (non-constant time)",
    [
        "Index clock τ = n (no constant-time assumption)",
        "Arc-length clock dτ = |dΣ| (progress-based)",
        "Trap-warped clock dτ = dt·(1 + λ·Z) (Sandy time)"
    ],
    index=2
)
lam = st.sidebar.slider("λ (warp strength)", 0.0, 10.0, 2.0, 0.1)

st.sidebar.header("Toy-3")
early_frac = st.sidebar.slider("Early fraction", 0.05, 1.0, 0.25, 0.05)
corner_th = st.sidebar.slider("Corner threshold", 0.60, 0.95, 0.85, 0.01)
corners = st.sidebar.multiselect("Corners to measure", ["UR", "UL", "LR", "LL"], default=["UR", "LR"])

# ==================================================
# Clean + preprocess
# ==================================================
t = pd.to_numeric(df[time_col], errors="coerce").astype(float).to_numpy()
f = pd.to_numeric(df[flux_col], errors="coerce").astype(float).to_numpy()

mask = np.isfinite(t) & np.isfinite(f)
t, f = t[mask], f[mask]
order = np.argsort(t)
t, f = t[order], f[order]

if len(t) < 10:
    st.error("Not enough valid points after cleaning.")
    st.stop()

if use_mag:
    f = 10 ** (-0.4 * f)

f_s = rolling_median(f, smooth_win)

# Σ = normalized smoothed flux (escape proxy)
Sigma = normalize_01(f_s)

# ==================================================
# Build Z from *change* (but not using constant time)
# Start with change-per-index as a neutral measure
# ==================================================
dSigma_dn = np.abs(np.gradient(Sigma))  # per-sample change
dSigma_dn_n = normalize_01(dSigma_dn)

# Neutral trap proxy: high Z when Σ changes slowly
Z_base = 1.0 - dSigma_dn_n
Z_base = np.clip(Z_base, 0, 1)

# ==================================================
# Build local time τ (NOT constant)
# ==================================================
n = np.arange(len(Sigma), dtype=float)
dt = np.gradient(t)  # observational spacing (may be uneven)

if tau_mode.startswith("Index clock"):
    tau = n

elif tau_mode.startswith("Arc-length clock"):
    # progress clock: accumulate motion in Σ (no external time)
    # add epsilon to avoid zero-steps
    eps = 1e-6
    d_tau = np.sqrt((np.gradient(Sigma) ** 2) + eps**2)
    tau = np.cumsum(d_tau)

else:
    # Sandy time: dτ = dt * (1 + λ*Z)
    # Use Z_base as structure field for warping
    d_tau = np.maximum(dt, 0) * (1.0 + lam * Z_base)
    # Ensure monotonic even if dt has issues
    d_tau = np.where(np.isfinite(d_tau), d_tau, 0.0)
    tau = np.cumsum(np.maximum(d_tau, 1e-9))

# ==================================================
# Recompute dynamics in τ-space
# ==================================================
dSigma_dtau = np.abs(safe_gradient(Sigma, tau))
dSigma_dtau_n = normalize_01(dSigma_dtau)

# Final Z: trap strength high when Σ changes slowly in local time
Z = 1.0 - dSigma_dtau_n
Z = np.clip(Z, 0, 1)

# Quench = max curvature in τ-space
curv = safe_gradient(safe_gradient(Sigma, tau), tau)
quench_idx = int(np.argmax(np.abs(curv)))
quench_tau = float(tau[quench_idx])

# Early window in sample-count (keeps “early” definition consistent)
N = len(tau)
N_early = max(10, int(np.floor(N * early_frac)))
tau_e = tau[:N_early]
Z_e = Z[:N_early]
S_e = Sigma[:N_early]

# ==================================================
# Diagnostics (in τ-space, not t)
# ==================================================
st.subheader("Diagnostics (computed in local time τ)")
c1, c2, c3 = st.columns(3)
c1.metric("Points", f"{N}")
c2.metric("Quench τ", f"{quench_tau:.4f}")
c3.metric("Early window points", f"{N_early}")

# Corner dwell per-corner (fraction of EARLY window)
total = 0.0
if corners:
    dcols = st.columns(len(corners))
    for i, c in enumerate(corners):
        m = corner_mask(Z_e, S_e, corner_th, c)
        frac = float(np.mean(m))
        total += frac
        dcols[i].metric(f"{c} dwell (early)", f"{100*frac:.2f}%")
else:
    st.info("Select corners in the sidebar to measure dwell.")

# Simple interpretation
if corners:
    if total < 0.03:
        st.success("Low dwell → early evolution not sticky (weak precursor).")
    elif total < 0.10:
        st.warning("Moderate dwell → early stickiness (precursor structure).")
    else:
        st.error("High dwell → strong corner locking (transition likely).")

# ==================================================
# Plots
# ==================================================
st.subheader("Plots (τ is the clock)")

# Show raw flux vs t just for reference (not used as “clock” unless you choose Sandy warp)
plot_series(t, {"flux_raw": f, "flux_smooth": f_s}, "Light curve (reference only)", "t (input coordinate)")

plot_series(tau, {"Σ": Sigma, "Z": Z}, "Derived Σ and Z vs local time τ", "τ (local time)")

cA, cB = st.columns(2)
with cA:
    plot_phase(Z, Sigma, "Phase space (full)")

with cB:
    plot_phase(Z_e, S_e, f"Phase space (early {early_frac:.0%})")

with st.expander("What changed (why this is correct for Sandy’s Law)"):
    st.markdown(
        """
### Why your objection was correct
If time is *not* constant, you cannot use **dΣ/dt** as a fundamental driver.

### What we do instead
We build a **local time τ** and compute **dΣ/dτ**.

- **Index clock τ=n**: no external time at all  
- **Arc-length clock**: time = accumulated progress in Σ  
- **Trap-warped clock**: τ evolves faster/slower depending on Z (structure)

This is exactly your Sandy’s Law claim: **time is a local variable controlled by structure**.
"""
    )