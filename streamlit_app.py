import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Sandy’s Law — Square (CSV)",
    layout="wide"
)

st.title("Sandy’s Law — Square (CSV)")
st.caption("Real early light curves → Corner Dwell → Quench")

# ============================================================
# SIDEBAR — INPUT
# ============================================================

st.sidebar.header("Input Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload light curve CSV",
    type=["csv"]
)

st.sidebar.markdown("---")
st.sidebar.header("Preprocessing")

is_magnitude = st.sidebar.checkbox(
    "Flux column is magnitude (convert to flux)",
    value=False
)

normalize = st.sidebar.checkbox(
    "Normalize flux",
    value=True
)

smooth_window = st.sidebar.slider(
    "Smoothing window (odd)",
    min_value=1,
    max_value=101,
    value=15,
    step=2
)

st.sidebar.markdown("---")
st.sidebar.header("Toy-3 Parameters")

z_th = st.sidebar.slider("Z threshold (trap)", 0.5, 0.95, 0.7, 0.01)
s_th = st.sidebar.slider("Σ threshold (escape)", 0.05, 0.5, 0.25, 0.01)

# ============================================================
# LOAD CSV
# ============================================================

if uploaded_file is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)

required_cols = {"time", "flux"}
if not required_cols.issubset(df.columns):
    st.error("CSV must contain columns: time, flux")
    st.stop()

df = df.sort_values("time")

time = df["time"].values.astype(float)
flux = df["flux"].values.astype(float)

# ============================================================
# MAG → FLUX (optional)
# ============================================================

if is_magnitude:
    flux = 10 ** (-0.4 * flux)

# ============================================================
# NORMALIZE & SMOOTH
# ============================================================

if normalize:
    flux = (flux - np.min(flux)) / (np.max(flux) - np.min(flux))

if smooth_window > 1:
    if smooth_window >= len(flux):
        st.error("Smoothing window too large for dataset.")
        st.stop()
    flux = savgol_filter(flux, smooth_window, polyorder=2)

# ============================================================
# SANDY’S LAW VARIABLES
# ============================================================

# Σ(t): escape proxy (rate of change)
Sigma = np.abs(np.gradient(flux, time))
Sigma = (Sigma - Sigma.min()) / (Sigma.max() - Sigma.min())

# Z(t): trap proxy (inverse activity)
Z = 1.0 - flux
Z = (Z - Z.min()) / (Z.max() - Z.min())

# ============================================================
# CORNER DWELL & QUENCH
# ============================================================

corner_mask = (Z > z_th) & (Sigma < s_th)

dt = np.median(np.diff(time))
corner_time = np.sum(corner_mask) * dt
corner_fraction = np.sum(corner_mask) / len(Z)

# quench = max curvature
d2 = np.gradient(np.gradient(Sigma, time), time)
quench_index = np.argmax(np.abs(d2))
quench_time = time[quench_index]

# ============================================================
# METRICS
# ============================================================

st.subheader("Diagnostics")

c1, c2, c3 = st.columns(3)
c1.metric("Corner dwell time", f"{corner_time:.3f}")
c2.metric("Corner dwell fraction", f"{corner_fraction*100:.2f}%")
c3.metric("Quench time", f"{quench_time:.3f}")

# ============================================================
# PLOTS
# ============================================================

st.subheader("Sandy’s Law — Square")

col1, col2, col3 = st.columns(3)

with col1:
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(Z, Sigma, lw=1.4)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xlabel("Z (Trap Strength)")
    ax.set_ylabel("Σ (Escape)")
    ax.set_title("Phase Space")
    ax.grid(alpha=0.4)
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(time, flux, lw=1.4)
    ax.axvline(quench_time, color="r", ls="--", label="Quench")
    ax.set_xlabel("Time")
    ax.set_ylabel("Flux")
    ax.set_title("Light Curve")
    ax.legend()
    ax.grid(alpha=0.4)
    st.pyplot(fig)

with col3:
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(time, corner_mask.astype(int), lw=1.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Corner State")
    ax.set_title("Corner Dwell Indicator")
    ax.grid(alpha=0.4)
    st.pyplot(fig)

# ============================================================
# INTERPRETATION
# ============================================================

with st.expander("What this means (Sandy’s Law)"):
    st.markdown(
        """
• **High Z + Low Σ** → trapped photon / energy state  
• **Corner dwell** → delayed escape before visible rise  
• **Quench** → structural release (shock breakout / diffusion escape)  

This diagnostic works **before peak brightness** and is
**independent of light-curve fitting models**.
"""
    )