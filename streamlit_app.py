import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(
    page_title="Sandy’s Law — Square Overlay → Shared Time",
    layout="wide"
)

st.title("Sandy’s Law — Square Tiling & Shared Time")
st.caption("Local independence → square traversal → overlap → Toy-3 locking")

# =====================================================
# INPUT (EVENTS ONLY)
# =====================================================
st.header("1️⃣ Event Input (No Time)")

csv_text = st.text_area(
    "Paste CSV (columns: index, flux)",
    height=220,
    placeholder="index,flux\n0,1.01\n1,1.02\n..."
)

uploaded = st.file_uploader("or upload CSV", type=["csv"])

if not csv_text.strip() and uploaded is None:
    st.info("Paste or upload CSV to begin.")
    st.stop()

try:
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_csv(StringIO(csv_text))
except Exception as e:
    st.error(f"CSV parse error: {e}")
    st.stop()

if "flux" not in [c.lower() for c in df.columns]:
    st.error("CSV must contain a 'flux' column")
    st.stop()

flux = df["flux"].astype(float).values
flux = flux[np.isfinite(flux)]

N = len(flux)
if N < 12:
    st.error("Not enough events")
    st.stop()

# =====================================================
# CONTROLS
# =====================================================
st.sidebar.header("Controls")

square_window = st.sidebar.slider(
    "Local square window (events)",
    4, 20, 6, 1
)

smooth = st.sidebar.slider(
    "Flux smoothing",
    1, 31, 7, 2
)

corner_th = st.sidebar.slider(
    "Toy-3 corner threshold",
    0.70, 0.95, 0.85, 0.01
)

# =====================================================
# Σ = ESCAPE STRUCTURE
# =====================================================
def normalize(x):
    lo, hi = np.min(x), np.max(x)
    if hi - lo < 1e-12:
        return np.full_like(x, 0.5)
    return (x - lo) / (hi - lo)

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
dSigma_dtau = np.gradient(Sigma, tau, edge_order=1)
Z = 1.0 - np.abs(dSigma_dtau)
Z = np.clip(Z, 0, 1)

# =====================================================
# LOCAL SQUARE CONSTRUCTION (RESTORED)
# =====================================================
# Squares = bounded traversal inside sliding windows

squares = []
for i in range(N - square_window):
    z_loc = Z[i:i+square_window]
    s_loc = Sigma[i:i+square_window]
    squares.append((z_loc, s_loc))

# =====================================================
# METRICS: SQUARE AREA & OVERLAP
# =====================================================

square_areas = []
square_boxes = []

for z_loc, s_loc in squares:
    dz = np.max(z_loc) - np.min(z_loc)
    ds = np.max(s_loc) - np.min(s_loc)
    area = dz * ds
    square_areas.append(area)

    # bounding box for overlap test
    square_boxes.append((
        np.min(z_loc), np.max(z_loc),
        np.min(s_loc), np.max(s_loc)
    ))

square_areas = np.array(square_areas)

# Mean square area
mean_square_area = np.mean(square_areas)

# -----------------------------------------------------
# Overlap fraction
# -----------------------------------------------------
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
# TOY-3 CORNER LOCKING
# =====================================================
in_corner = (Z > corner_th) & (Sigma > corner_th)

shared_frac = in_corner.mean()

# =====================================================
# PLOTS
# =====================================================
st.header("2️⃣ Phase Geometry")

fig, ax = plt.subplots(figsize=(6,6))

# Plot all square loops
for z_loc, s_loc in squares:
    ax.plot(z_loc, s_loc, color="gray", alpha=0.35, lw=1)

# Global trajectory
ax.plot(Z, Sigma, color="black", lw=1.8, label="Global path")

# Corner dwell
ax.scatter(
    Z[in_corner],
    Sigma[in_corner],
    c="red",
    s=14,
    label="Shared-time dwell"
)

ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_aspect("equal")
ax.set_xlabel("Z (trap strength)")
ax.set_ylabel("Σ (escape)")
ax.set_title("Square Tiling → Overlap → Shared Time")
ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)
plt.close(fig)

# =====================================================
# DIAGNOSTICS
# =====================================================
st.header("3️⃣ Diagnostics")

c1, c2 = st.columns(2)

c1.metric("Square count", len(squares))
c2.metric("Shared-time fraction", f"{100*shared_frac:.2f}%")

if shared_frac < 0.05:
    st.success("Independent regime — square tiling only")
elif shared_frac < 0.15:
    st.warning("Square overlap — coupling emerging")
else:
    st.error("Shared-time regime — Toy-3 locking")

# =====================================================
# INTERPRETATION
# =====================================================
with st.expander("Interpretation (RESTORED)"):
    st.markdown(
        """
**What you are seeing:**

• Each gray loop is a **local square**  
• Squares represent **independent causal traversal**  
• Many squares can occupy the **same global region**  
• Overlap destroys independence  
• Corner locking = **shared time**

This restores the photon-pathway logic:
local paths → overlap → coordination → release.
"""
    )