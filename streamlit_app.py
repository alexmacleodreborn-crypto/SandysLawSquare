import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(
    page_title="Sandy’s Law — Square Tiling → Shared Time",
    layout="wide"
)

st.title("Sandy’s Law — Square Tiling, Exhaust & Shared Time")
st.caption("Event structure only • No flowing time • Photon-domain instrument")

# =====================================================
# INPUT
# =====================================================
st.header("1️⃣ Input: Event List (No Time)")

csv_text = st.text_area(
    "Paste CSV (required column: flux)",
    height=220,
    placeholder="index,flux\n0,1.01\n1,1.02\n2,1.01\n..."
)

uploaded = st.file_uploader("or upload CSV", type=["csv"])

if not csv_text.strip() and uploaded is None:
    st.info("Paste or upload event data to begin.")
    st.stop()

try:
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_csv(StringIO(csv_text))
except Exception as e:
    st.error(f"CSV parse error: {e}")
    st.stop()

if "flux" not in [c.lower() for c in df.columns]:
    st.error("CSV must contain a 'flux' column")
    st.stop()

flux = df[df.columns[[c.lower() for c in df.columns].index("flux")]].astype(float).values
flux = flux[np.isfinite(flux)]

N = len(flux)
if N < 12:
    st.error("Not enough events to form squares.")
    st.stop()

# =====================================================
# CONTROLS
# =====================================================
st.sidebar.header("Controls")

square_window = st.sidebar.slider("Square window (events)", 4, 20, 6, 1)
smooth = st.sidebar.slider("Flux smoothing", 1, 31, 7, 2)
corner_th = st.sidebar.slider("Corner threshold", 0.70, 0.95, 0.85, 0.01)

A_th = st.sidebar.slider(
    "A_th (independence area threshold)",
    0.01, 0.90, 0.25, 0.01
)

C_th = st.sidebar.slider(
    "C_th (shared-time threshold)",
    0.01, 0.50, 0.10, 0.01
)

# =====================================================
# Σ = ESCAPE STRUCTURE
# =====================================================
def normalize(x):
    lo, hi = np.min(x), np.max(x)
    return np.zeros_like(x) + 0.5 if hi - lo < 1e-12 else (x - lo) / (hi - lo)

if smooth > 1:
    flux_s = (
        pd.Series(flux)
        .rolling(smooth, center=True, min_periods=1)
        .median()
        .values
    )
else:
    flux_s = flux.copy()

Sigma = normalize(flux_s)

# =====================================================
# τ = STRUCTURAL COORDINATE (NOT TIME)
# =====================================================
dSigma = np.abs(np.diff(Sigma, prepend=Sigma[0]))
tau = np.cumsum(dSigma)
tau = tau / np.max(tau)

# =====================================================
# Z = TRAP STRENGTH
# =====================================================
Z = 1.0 - np.abs(np.gradient(Sigma, tau, edge_order=1))
Z = np.clip(Z, 0.0, 1.0)

# =====================================================
# SQUARE CONSTRUCTION
# =====================================================
squares = []
square_boxes = []
square_areas = []

for i in range(N - square_window):
    z_loc = Z[i:i + square_window]
    s_loc = Sigma[i:i + square_window]

    dz = np.max(z_loc) - np.min(z_loc)
    ds = np.max(s_loc) - np.min(s_loc)
    area = dz * ds

    squares.append((z_loc, s_loc))
    square_areas.append(area)

    square_boxes.append((
        np.min(z_loc), np.max(z_loc),
        np.min(s_loc), np.max(s_loc)
    ))

square_areas = np.array(square_areas)
mean_square_area = float(np.mean(square_areas))

# =====================================================
# OVERLAP (EXHAUST SATURATION)
# =====================================================
overlap_count = 0
total_pairs = 0

for i in range(len(square_boxes)):
    z1_min, z1_max, s1_min, s1_max = square_boxes[i]
    for j in range(i + 1, len(square_boxes)):
        z2_min, z2_max, s2_min, s2_max = square_boxes[j]

        overlap = (
            z1_min <= z2_max and z2_min <= z1_max and
            s1_min <= s2_max and s2_min <= s1_max
        )

        if overlap:
            overlap_count += 1
        total_pairs += 1

overlap_fraction = overlap_count / total_pairs if total_pairs > 0 else 0.0

# =====================================================
# TOY-3 CORNER LOCK (SHARED TIME)
# =====================================================
in_corner = (Z > corner_th) & (Sigma > corner_th)
shared_fraction = in_corner.mean()

# =====================================================
# PLOT
# =====================================================
st.header("2️⃣ Phase Geometry")

fig, ax = plt.subplots(figsize=(6, 6))

for z_loc, s_loc in squares:
    ax.plot(z_loc, s_loc, color="gray", alpha=0.35, lw=1)

ax.plot(Z, Sigma, color="black", lw=1.8, label="Global path")
ax.scatter(Z[in_corner], Sigma[in_corner], c="red", s=16, label="Shared-time dwell")

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.set_xlabel("Z (trap strength)")
ax.set_ylabel("Σ (escape)")
ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)
plt.close(fig)

# =====================================================
# DIAGNOSTICS
# =====================================================
st.header("3️⃣ Diagnostics")

c1, c2, c3 = st.columns(3)
c1.metric("Square count", len(square_areas))
c2.metric("Mean square area", f"{mean_square_area:.4f}")
c3.metric("Overlap fraction", f"{100 * overlap_fraction:.2f}%")

st.metric("Shared-time fraction", f"{100 * shared_fraction:.2f}%")

# =====================================================
# REGIME CLASSIFICATION (FIXED)
# =====================================================
st.header("4️⃣ Regime Classification")

if shared_fraction >= C_th:
    st.error("Shared-time regime — phase locked")
elif mean_square_area <= A_th and overlap_fraction > 0.1:
    st.warning("Coupled regime — exhaust saturating")
else:
    st.success("Independent regime — exhaust free")

# =====================================================
# INTERPRETATION
# =====================================================
with st.expander("Interpretation (Locked)"):
    st.markdown(
        """
• **Square area** = independence / exhaust capacity  
• **Overlap** = reuse of structure (not failure by itself)  
• **Shared-time fraction** = phase lock  

Exhaust saturation requires **small area + high overlap**.  
Shared time requires **sustained corner dwell**.

Time does not flow here.
Time is a **phase label**.
"""
    )