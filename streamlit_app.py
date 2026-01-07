import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(
    page_title="Sandy’s Law — Event Geometry (Toy 3)",
    layout="wide"
)

st.title("Sandy’s Law — Event Geometry & Shared Time")
st.caption("No global time • No flow • Ordering emerges from structure")

# =====================================================
# INPUT: EVENTS ONLY
# =====================================================
st.header("1️⃣ Input: Event List (No Time)")

st.markdown(
    """
The input is **not a light curve in time**.

It is an **ordered list of emission events**.
The index is only a label — physics does **not** use it.
"""
)

csv_text = st.text_area(
    "Paste CSV (required columns: index, flux)",
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

# Validate
cols = [c.lower() for c in df.columns]
if "flux" not in cols:
    st.error("CSV must contain a 'flux' column")
    st.stop()

flux = df[df.columns[cols.index("flux")]].astype(float).values
flux = flux[np.isfinite(flux)]

if len(flux) < 10:
    st.error("Not enough events.")
    st.stop()

# =====================================================
# CONTROLS
# =====================================================
st.sidebar.header("Controls")

smooth = st.sidebar.slider("Flux smoothing (structure only)", 1, 31, 9, 2)
corner_th = st.sidebar.slider("Shared-time corner threshold", 0.70, 0.95, 0.85, 0.01)

# =====================================================
# Σ = ESCAPE (STRUCTURE ONLY)
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
# τ = EMERGENT COORDINATE (NOT TIME FLOW)
# =====================================================
# τ is an ordering coordinate induced by structural change
dSigma = np.abs(np.diff(Sigma, prepend=Sigma[0]))
tau = np.cumsum(dSigma)

if np.max(tau) > 0:
    tau = tau / np.max(tau)

# =====================================================
# Z = TRAP STRENGTH (GEOMETRIC)
# =====================================================
# Resistance to change in Σ along τ
dSigma_dtau = np.gradient(Sigma, tau, edge_order=1)
Z = 1.0 - np.abs(dSigma_dtau)
Z = np.clip(Z, 0.0, 1.0)

# =====================================================
# TOY 3: SHARED-TIME GEOMETRY
# =====================================================
in_corner = (Z > corner_th) & (Sigma > corner_th)

tags = np.zeros(len(Z), dtype=int)
for i in range(len(Z)):
    frac = in_corner[: i + 1].mean()
    if frac < 0.03:
        tags[i] = 0      # Independent
    elif frac < 0.10:
        tags[i] = 1      # Coupled
    else:
        tags[i] = 2      # Shared

shared_idx = np.where(tags == 2)[0]
shared_tau = tau[shared_idx[0]] if len(shared_idx) else None

# =====================================================
# DIAGNOSTICS
# =====================================================
st.header("2️⃣ Regime Diagnostics")

c1, c2 = st.columns(2)

c1.metric("Shared-time fraction", f"{100*in_corner.mean():.2f}%")
c2.metric("Shared-time onset (τ)", "—" if shared_tau is None else f"{shared_tau:.3f}")

if shared_tau is None:
    st.success("Independent regime (no shared time)")
elif in_corner.mean() < 0.15:
    st.warning("Partial coupling (incipient shared time)")
else:
    st.error("Shared-time regime (Toy-3 corner locking)")

# =====================================================
# PLOTS (GEOMETRIC, NOT TEMPORAL)
# =====================================================
st.header("3️⃣ Geometric Plots")

# Flux vs event index (label only)
fig, ax = plt.subplots(figsize=(8,3))
ax.plot(flux, lw=1.2)
ax.set_xlabel("event index (label only)")
ax.set_ylabel("flux")
ax.set_title("Event Flux (no time implied)")
ax.grid(alpha=0.3)
st.pyplot(fig); plt.close(fig)

# Phase space
fig, ax = plt.subplots(figsize=(6,6))
ax.plot(Z, Sigma, lw=1.2)
ax.scatter(Z[in_corner], Sigma[in_corner], c="red", s=10, label="shared-time dwell")
ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.set_aspect("equal")
ax.set_xlabel("Z (trap strength)")
ax.set_ylabel("Σ (escape)")
ax.set_title("Phase Geometry (Toy 3)")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig); plt.close(fig)

# Tag geometry
fig, ax = plt.subplots(figsize=(8,2.5))
ax.plot(tags, drawstyle="steps-post")
ax.set_yticks([0,1,2])
ax.set_yticklabels(["Independent","Coupled","Shared"])
ax.set_xlabel("event index")
ax.set_title("Emergence of Shared Time (no flow)")
ax.grid(alpha=0.3)
st.pyplot(fig); plt.close(fig)

# =====================================================
# INTERPRETATION
# =====================================================
with st.expander("Interpretation (Locked)"):
    st.markdown(
        """
• There is **no time variable** in this model  
• Events are **not evolving** — they are **related**  
• τ is a **coordinate of structure**, not duration  
• Shared time is **detected**, not assumed  
• Toy 3 is **pure geometry**, not dynamics  

Nothing here flows.
Only relationships exist.
"""
    )