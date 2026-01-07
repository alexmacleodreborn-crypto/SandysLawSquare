import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(
    page_title="Sandy’s Law — Phase Coherence (No Time)",
    layout="wide"
)

st.title("Sandy’s Law — Phase Coherence Instrument")
st.caption("Phase geometry only • No time • Coherence C ∈ [0,1]")

# =====================================================
# INPUT
# =====================================================
st.header("1️⃣ Paste Phase Event CSV")

st.markdown(
    """
**Required columns**:
- `z` — trap strength proxy ∈ [0,1]
- `sigma` — escape proxy ∈ [0,1]

Each row = one event.  
Order is **not time** and carries no physics.
"""
)

csv_text = st.text_area(
    "Paste CSV",
    height=260,
    placeholder="event_id,z,sigma\n1,0.12,0.18\n2,0.14,0.21\n..."
)

if not csv_text.strip():
    st.info("Paste a CSV to begin.")
    st.stop()

# =====================================================
# CSV PARSER (STRICT + SAFE)
# =====================================================
try:
    df = pd.read_csv(StringIO(csv_text))
except Exception as e:
    st.error(f"CSV parse error: {e}")
    st.stop()

df.columns = [c.strip().lower() for c in df.columns]

if "z" not in df.columns or "sigma" not in df.columns:
    st.error("CSV must contain columns: z, sigma")
    st.stop()

Z = pd.to_numeric(df["z"], errors="coerce").to_numpy()
S = pd.to_numeric(df["sigma"], errors="coerce").to_numpy()

mask = np.isfinite(Z) & np.isfinite(S)
Z, S = Z[mask], S[mask]

N = len(Z)
if N < 8:
    st.error("At least 8 events required for coherence analysis.")
    st.stop()

# =====================================================
# CONTROLS
# =====================================================
st.sidebar.header("Phase Tiling")

bins = st.sidebar.slider(
    "Squares per axis",
    6, 80, 24, 2
)

min_shared = st.sidebar.slider(
    "Minimum shared events per square",
    2, 8, 2, 1
)

# =====================================================
# PHASE TILING
# =====================================================
zi = np.clip((Z * bins).astype(int), 0, bins - 1)
si = np.clip((S * bins).astype(int), 0, bins - 1)

occ = np.zeros((bins, bins), dtype=int)
for a, b in zip(zi, si):
    occ[b, a] += 1  # row = sigma, col = z

shared_mask = occ >= min_shared

shared_events = 0
for a, b in zip(zi, si):
    if shared_mask[b, a]:
        shared_events += 1

C = shared_events / N

# =====================================================
# VISUALISATION
# =====================================================
st.header("2️⃣ Phase Geometry")

fig, ax = plt.subplots(figsize=(6, 6))

ax.scatter(Z, S, s=30, alpha=0.8, label="Events")

ys, xs = np.where(shared_mask)
if len(xs) > 0:
    cx = (xs + 0.5) / bins
    cy = (ys + 0.5) / bins
    ax.scatter(
        cx, cy,
        s=120,
        marker="s",
        facecolors="none",
        edgecolors="red",
        linewidths=1.5,
        label="Shared phase squares"
    )

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.set_xlabel("Z (trap)")
ax.set_ylabel("Σ (escape)")
ax.grid(alpha=0.3)
ax.legend()

st.pyplot(fig)
plt.close(fig)

# =====================================================
# DIAGNOSTICS
# =====================================================
st.header("3️⃣ Coherence Diagnostics")

c1, c2, c3 = st.columns(3)
c1.metric("Event count", N)
c2.metric("Shared events", shared_events)
c3.metric("Coherence C", f"{C:.3f}")

# =====================================================
# REGIME INTERPRETATION
# =====================================================
st.header("4️⃣ Regime")

if C == 0:
    st.success("Independent regime (C = 0)")
elif C < 0.3:
    st.warning("Weak coherence")
elif C < 0.7:
    st.warning("Partial macroscopic coherence")
else:
    st.error("Strong macroscopic coherence (C → 1)")

# =====================================================
# INTERPRETATION
# =====================================================
with st.expander("Interpretation (Locked)"):
    st.markdown(
        """
- This instrument **does not use time**.
- Coherence arises from **phase crowding**, not simultaneity.
- C measures the fraction of events forced into shared phase windows.
- Time-like behaviour is **emergent**, not assumed.

This obeys Sandy’s Law: **Trap → Transition → Escape**.
"""
    )