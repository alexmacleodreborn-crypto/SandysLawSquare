import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

# =====================================================
# App config
# =====================================================
st.set_page_config(
    page_title="Sandy’s Law — Emergent Time (Toy 3)",
    layout="wide"
)

st.title("Sandy’s Law — Emergent Time from Flux Structure")
st.caption("Input: flux only | τ, Z, Σ are derived | No global clock")

# =====================================================
# CSV INPUT (OBSERVABLES ONLY)
# =====================================================
st.header("1️⃣ Input: Light Curve (Flux Only)")

csv_text = st.text_area(
    "Paste CSV (must contain columns: time, flux)",
    height=220,
    placeholder="time,flux\n0.0,1.01\n0.1,1.02\n..."
)

if not csv_text.strip():
    st.info("Paste CSV data to begin.")
    st.stop()

try:
    df = pd.read_csv(StringIO(csv_text))
except Exception as e:
    st.error(f"CSV parse error: {e}")
    st.stop()

# Validate columns
cols = [c.lower() for c in df.columns]
if "time" not in cols or "flux" not in cols:
    st.error("CSV must contain columns named 'time' and 'flux'")
    st.stop()

t = df[df.columns[cols.index("time")]].astype(float).values
flux = df[df.columns[cols.index("flux")]].astype(float).values

# Clean
mask = np.isfinite(t) & np.isfinite(flux)
t, flux = t[mask], flux[mask]

# Order by observation order only
order = np.argsort(t)
t, flux = t[order], flux[order]

if len(flux) < 10:
    st.error("Not enough data points.")
    st.stop()

# =====================================================
# SIDEBAR CONTROLS
# =====================================================
st.sidebar.header("Processing")

smooth_win = st.sidebar.slider("Flux smoothing window", 1, 51, 9, 2)

corner_th = st.sidebar.slider(
    "Corner threshold (Toy 3)",
    0.60, 0.95, 0.85, 0.01
)

early_frac = st.sidebar.slider(
    "Early fraction (diagnostics)",
    0.05, 1.0, 0.30, 0.05
)

# =====================================================
# Σ = ESCAPE (FROM FLUX)
# =====================================================
def normalize_01(x):
    lo, hi = np.min(x), np.max(x)
    if hi - lo < 1e-12:
        return np.full_like(x, 0.5)
    return (x - lo) / (hi - lo)

if smooth_win > 1:
    flux_s = (
        pd.Series(flux)
        .rolling(smooth_win, center=True, min_periods=1)
        .median()
        .values
    )
else:
    flux_s = flux.copy()

Sigma = normalize_01(flux_s)

# =====================================================
# τ = EMERGENT LOCAL TIME (STRUCTURE-BASED)
# =====================================================
# τ advances when Σ changes, not when clock ticks
dSigma = np.abs(np.diff(Sigma, prepend=Sigma[0]))
tau = np.cumsum(dSigma)

if np.max(tau) > 0:
    tau = tau / np.max(tau)

# =====================================================
# Z = TRAP STRENGTH (SLOW CHANGE IN Σ PER τ)
# =====================================================
# Safe gradient in τ-space
dSigma_dtau = np.gradient(Sigma, tau, edge_order=1)
Z = 1.0 - np.abs(dSigma_dtau)
Z = np.clip(Z, 0.0, 1.0)

# =====================================================
# TOY 3: CORNER DWELL + TAGGING
# =====================================================
in_corner = (Z > corner_th) & (Sigma > corner_th)
dwell_fraction = in_corner.mean()

# Regime tags
# 0 = local / independent
# 1 = coupled
# 2 = shared time
tags = np.zeros(len(Z), dtype=int)

for i in range(len(Z)):
    frac = in_corner[: i + 1].mean()
    if frac < 0.03:
        tags[i] = 0
    elif frac < 0.10:
        tags[i] = 1
    else:
        tags[i] = 2

shared_idx = np.where(tags == 2)[0]
shared_tau = tau[shared_idx[0]] if len(shared_idx) else None

# Early window
N = len(Z)
N_early = max(10, int(N * early_frac))
Z_e = Z[:N_early]
S_e = Sigma[:N_early]
tau_e = tau[:N_early]

# =====================================================
# OUTPUT: DIAGNOSTICS
# =====================================================
st.header("2️⃣ Diagnostics")

c1, c2, c3 = st.columns(3)
c1.metric("Early dwell fraction", f"{100 * ((Z_e > corner_th) & (S_e > corner_th)).mean():.2f}%")
c2.metric("Shared-time onset τ", "—" if shared_tau is None else f"{shared_tau:.3f}")
c3.metric("Total points", len(Z))

if dwell_fraction < 0.05:
    st.success("Independent regime (no shared time)")
elif dwell_fraction < 0.15:
    st.warning("Coupled regime (shared time forming)")
else:
    st.error("Shared-time regime (Toy-3 corner locking)")

# =====================================================
# PLOTS
# =====================================================
st.header("3️⃣ Plots")

# Light curve (reference only)
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(t, flux, alpha=0.5, label="raw flux")
ax.plot(t, flux_s, lw=1.2, label="smoothed flux")
ax.set_xlabel("observation order (time)")
ax.set_ylabel("flux")
ax.set_title("Light Curve (reference only — not the clock)")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)
plt.close(fig)

# Σ and Z vs τ
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(tau, Sigma, label="Σ (escape)")
ax.plot(tau, Z, label="Z (trap)")
ax.set_xlabel("τ (emergent local time)")
ax.set_ylabel("value")
ax.set_title("Derived Variables vs Emergent Time τ")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)
plt.close(fig)

# Phase space
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(Z, Sigma, lw=1.2)
ax.scatter(Z[in_corner], Sigma[in_corner], c="red", s=8, label="corner dwell")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.set_xlabel("Z (trap strength)")
ax.set_ylabel("Σ (escape)")
ax.set_title("Phase Space (Toy 3)")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)
plt.close(fig)

# Tag timeline
fig, ax = plt.subplots(figsize=(8, 2.5))
ax.plot(tau, tags, drawstyle="steps-post")
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(["Local", "Coupled", "Shared"])
ax.set_xlabel("τ")
ax.set_title("Emergence of Shared Time")
ax.grid(alpha=0.3)
st.pyplot(fig)
plt.close(fig)

# =====================================================
# INTERPRETATION
# =====================================================
with st.expander("Physical Interpretation (Locked)"):
    st.markdown(
        """
**What this file enforces:**

• CSV contains **only observables**  
• **Time is not assumed** — τ emerges from flux structure  
• Z and Σ are **derived**, not fitted  
• Corner dwell = **shared-time formation**  
• Toy 3 is a **causal template**, not a simulation  

This directly matches your photon-pathway explanation:
independent events → interaction → shared clock → release.
"""
    )