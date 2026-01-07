import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==================================================
# App config
# ==================================================
st.set_page_config(
    page_title="Sandy’s Law — Square (Toy 3 Mapping)",
    layout="wide"
)

st.title("Sandy’s Law — Map Light Curve → Toy 3 (Z, Σ)")
st.caption("CSV → Early light curve → Corner dwell diagnostics")

# ==================================================
# Utility functions
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

def corner_mask(Z, S, th, corner):
    hi = th
    lo = 1 - th
    if corner == "UR":
        return (Z >= hi) & (S >= hi)
    if corner == "UL":
        return (Z <= lo) & (S >= hi)
    if corner == "LR":
        return (Z >= hi) & (S <= lo)
    if corner == "LL":
        return (Z <= lo) & (S <= lo)
    return np.zeros_like(Z, dtype=bool)

def plot_phase(Z, S, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(Z, S, lw=1.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Z (Trap Strength)")
    ax.set_ylabel("Σ (Entropy Escape)")
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
    ax.plot(t, f, alpha=0.6, label="raw")
    ax.plot(t, fs, lw=1.2, label="smoothed")
    ax.set_xlabel("time")
    ax.set_ylabel("flux")
    ax.legend()
    ax.grid(alpha=0.35)
    st.pyplot(fig)
    plt.close(fig)

# ==================================================
# Sidebar
# ==================================================
st.sidebar.header("Load CSV")
uploaded = st.sidebar.file_uploader("Upload light curve CSV", type=["csv"])

st.sidebar.header("Preprocess")
smooth = st.sidebar.slider("Smoothing window", 1, 51, 11, 2)

st.sidebar.header("Toy-3 Mapping")
z_mode = st.sidebar.radio(
    "Z definition",
    [
        "Z = 1 − |dΣ/dt|  (trap = stability)",
        "Z = |dΣ/dt|      (trap = rapid change)"
    ]
)

early_frac = st.sidebar.slider("Early fraction", 0.05, 1.0, 0.25, 0.05)
corner_th = st.sidebar.slider("Corner threshold", 0.6, 0.95, 0.85, 0.01)

corners = st.sidebar.multiselect(
    "Corners to measure",
    ["UR", "UL", "LR", "LL"],
    default=["UR", "LR"]
)

# ==================================================
# Main
# ==================================================
if uploaded is None:
    st.info("Upload the example CSV below to begin.")
    st.stop()

df = pd.read_csv(uploaded)

time_col = st.selectbox("Time column", df.columns, index=0)
flux_col = st.selectbox("Flux column", df.columns, index=1)

t = df[time_col].to_numpy(dtype=float)
f = df[flux_col].to_numpy(dtype=float)

mask = np.isfinite(t) & np.isfinite(f)
t, f = t[mask], f[mask]

order = np.argsort(t)
t, f = t[order], f[order]

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

st.subheader("Derived Time Series")
plot_timeseries(t, Z, Sigma)

c1, c2 = st.columns(2)
with c1:
    plot_phase(Z, Sigma, "Phase Space (Full)")
with c2:
    plot_phase(Z_e, S_e, "Phase Space (Early)")

# ==================================================
# Corner diagnostics
# ==================================================
st.subheader("🔴 Corner Dwell Diagnostics (Toy 3)")

total = 0.0
cols = st.columns(len(corners) if corners else 1)

for i, c in enumerate(corners):
    mask_c = corner_mask(Z_e, S_e, corner_th, c)
    frac = mask_c.mean()
    total += frac
    cols[i].metric(f"{c} dwell", f"{100*frac:.2f}%")

if total < 0.03:
    st.success("Low dwell → smooth early evolution")
elif total < 0.10:
    st.warning("Moderate dwell → precursor structure")
else:
    st.error("High dwell → strong Toy-3 corner locking")

# ==================================================
# Interpretation
# ==================================================
with st.expander("Physical meaning (Sandy’s Law)"):
    st.markdown(
        """
**Σ (entropy escape)** is the observable signal (flux).  
**Z (trap strength)** is how constrained that signal is.

**Toy 3 statement:**  
> Systems approaching transition spend *time stuck in corners* of phase space.

Early-time **corner dwell** is a measurable precursor.
"""
    )