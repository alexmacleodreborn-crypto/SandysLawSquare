import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

# =====================================================
# App config
# =====================================================
st.set_page_config(
    page_title="Sandy’s Law — Event-Based Square Mapping",
    layout="wide"
)

st.title("Sandy’s Law — Event-Based Square Mapping")
st.caption("Event time ≠ constant time | Shared-time emergence | Toy 3 overlay")

# =====================================================
# CSV INPUT
# =====================================================
st.header("1️⃣ Paste CSV Lightcurve Data")

csv_text = st.text_area(
    "Paste CSV here (must contain time + flux columns)",
    height=220,
    placeholder="time,flux\n0.0,1.02\n0.1,0.98\n..."
)

if not csv_text.strip():
    st.stop()

# =====================================================
# LOAD CSV (FIXED)
# =====================================================
try:
    df = pd.read_csv(StringIO(csv_text))
except Exception as e:
    st.error(f"CSV parse error: {e}")
    st.stop()

# Column detection
cols = [c.lower() for c in df.columns]

if "time" not in cols or "flux" not in cols:
    st.error("CSV must contain 'time' and 'flux' columns")
    st.stop()

time = df[df.columns[cols.index("time")]].values
flux = df[df.columns[cols.index("flux")]].values

# Clean
mask = np.isfinite(time) & np.isfinite(flux)
time = time[mask]
flux = flux[mask]

# =====================================================
# EVENT TIME (NOT CONSTANT)
# =====================================================
st.header("2️⃣ Event Time Construction")

st.markdown(
    """
Time here is **event-derived**, not uniform.
We compute time **from flux changes**, not clock ticks.
"""
)

# Event time = cumulative flux gradient
d_flux = np.abs(np.diff(flux, prepend=flux[0]))
event_time = np.cumsum(d_flux)

# Normalize
event_time /= np.max(event_time)

# =====================================================
# MAP TO SANDY’S LAW VARIABLES
# =====================================================
st.header("3️⃣ Sandy’s Law Variable Mapping")

Z = (flux - flux.min()) / (flux.max() - flux.min())
Sigma = (event_time - event_time.min()) / (event_time.max() - event_time.min())

# =====================================================
# TOY 3: CORNER DWELL → QUENCH
# =====================================================
st.header("4️⃣ Toy 3 — Corner Dwell Overlay")

corner_threshold = st.slider("Corner Threshold", 0.7, 0.95, 0.85, 0.01)

in_corner = (
    ((Z > corner_threshold) & (Sigma > corner_threshold)) |
    ((Z < 1 - corner_threshold) & (Sigma < 1 - corner_threshold))
)

corner_fraction = np.mean(in_corner)

# =====================================================
# PLOTS
# =====================================================
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(Z, Sigma, lw=1)
    ax.scatter(Z[in_corner], Sigma[in_corner], c="red", s=6)
    ax.set_xlabel("Z (trap strength)")
    ax.set_ylabel("Σ (entropy escape)")
    ax.set_title("Phase Space (Toy 3 Overlay)")
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_aspect("equal")
    ax.grid(alpha=0.4)
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(event_time, Z, label="Z")
    ax.plot(event_time, Sigma, label="Σ")
    ax.set_xlabel("Event Time")
    ax.set_ylabel("Value")
    ax.set_title("Event-Time Series")
    ax.legend()
    ax.grid(alpha=0.4)
    st.pyplot(fig)

# =====================================================
# DIAGNOSTICS
# =====================================================
st.header("5️⃣ Corner Diagnostics")

st.metric("Corner dwell fraction", f"{corner_fraction*100:.2f}%")

if corner_fraction < 0.05:
    st.success("Stable regime — no imminent release")
elif corner_fraction < 0.15:
    st.warning("Pre-instability — shared-time clustering emerging")
else:
    st.error("Critical — collapse / release regime")

# =====================================================
# INTERPRETATION
# =====================================================
with st.expander("Physical Interpretation (Sandy’s Law)"):
    st.markdown(
        """
• **Time is event-derived**, not uniform  
• **Squares emerge from independent event clocks**  
• **Corner dwell = shared-time convergence**  
• **Release occurs when events synchronize in proximity**  

This matches early light-curve behavior **before peak luminosity**.
"""
    )