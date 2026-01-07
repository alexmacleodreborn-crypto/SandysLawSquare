import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(
    page_title="Sandy’s Law — Emergent Time (Toy 3)",
    layout="wide"
)

st.title("Sandy’s Law — Emergent Time from Structure")
st.caption("CSV time ≠ clock | τ emerges from Σ | Toy-3 structural overlay")

# =====================================================
# CSV INPUT (OBSERVABLES ONLY)
# =====================================================
st.header("1️⃣ Input CSV (Observables Only)")

csv_text = st.text_area(
    "Paste CSV (required columns: time, flux)",
    height=220,
    placeholder="time,flux\n0.0,1.01\n0.1,1.02\n..."
)

uploaded = st.file_uploader("or upload CSV", type=["csv"])

if not csv_text.strip() and uploaded is None:
    st.info("Paste or upload CSV to begin.")
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
if "time" not in cols or "flux" not in cols:
    st.error("CSV must contain columns: time, flux")
    st.stop()

t = df[df.columns[cols.index("time")]].astype(float).values
flux = df[df.columns[cols.index("flux")]].astype(float).values

mask = np.isfinite(t) & np.isfinite(flux)
t, flux = t[mask], flux[mask]

order = np.argsort(t)
t, flux = t[order], flux[order]

if len(flux) < 10:
    st.error("Not enough data points.")
    st.stop()

# =====================================================
# SIDEBAR CONTROLS
# =====================================================
st.sidebar.header("Controls")

smooth = st.sidebar.slider("Flux smoothing (median)", 1, 31, 9, 2)
corner_th = st.sidebar.slider("Toy-3 corner threshold", 0.70, 0.95, 0.85, 0.01)
early_frac = st.sidebar.slider("Early fraction", 0.10, 0.60, 0.30, 0.05)

# =====================================================
# Σ = ESCAPE (FROM FLUX)
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
# τ = EMERGENT LOCAL TIME (STRUCTURE-BASED)
# =====================================================
# τ advances ONLY when Σ changes
dSigma = np.abs(np.diff(Sigma, prepend=Sigma[0]))
tau = np.cumsum(dSigma)
tau = tau / np.max(tau)

# =====================================================
# Z = TRAP STRENGTH (SLOW Σ CHANGE IN τ)
# =====================================================
dSigma_dtau = np.gradient(Sigma, tau, edge_order=1)
Z = 1.0 - np.abs(dSigma_dtau)
Z = np.clip(Z, 0, 1)

# =====================================================
# TOY-3: CORNER DWELL + TAGGING
# =====================================================
in_corner = (Z > corner_th) & (Sigma > corner_th)

tags = np.zeros(len(Z), dtype=int)
for i in range(len(Z)):
    frac = in_corner[: i + 1].mean()
    if frac < 0.03:
        tags[i] = 0      # Local
    elif frac < 0.10:
        tags[i] = 1      # Coupled
    else:
        tags[i] = 2      # Shared

shared_idx = np.where(tags == 2)[0]
shared_tau = tau[shared_idx[0]] if len(shared_idx) else None

# Early window
N = len(Z)
N_e = max(10, int(N * early_frac))
Z_e, S_e = Z[:N_e], Sigma[:N_e]

# =====================================================
# DIAGNOSTICS
# =====================================================
st.header("2️⃣ Diagnostics")

c1, c2, c3 = st.columns(3)

c1.metric("Early dwell %",
          f"{100*((Z_e>corner_th)&(S_e>corner_th)).mean():.2f}%")

c2.metric("Shared-time onset τ",
          "—" if shared_tau is None else f"{shared_tau:.3f}")

c3.metric("Total events", len(Z))

if shared_tau is None:
    st.success("Independent regime (no shared time)")
elif shared_tau < tau[int(0.4*len(tau))]:
    st.warning("Early shared time (strong coupling)")
else:
    st.error("Late shared time (delayed release)")

# =====================================================
# PLOTS
# =====================================================
st.header("3️⃣ Plots")

# Reference light curve (NOT a clock)
fig, ax = plt.subplots(figsize=(8,3))
ax.plot(t, flux, alpha=0.5, label="raw flux")
ax.plot(t, flux_s, lw=1.2, label="smoothed flux")
ax.set_title("Light Curve (reference only)")
ax.set_xlabel("observation order")
ax.set_ylabel("flux")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig); plt.close(fig)

# Σ and Z vs τ
fig, ax = plt.subplots(figsize=(8,3))
ax.plot(tau, Sigma, label="Σ (escape)")
ax.plot(tau, Z, label="Z (trap)")
ax.set_title("Derived variables vs emergent τ")
ax.set_xlabel("τ (local time)")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig); plt.close(fig)

# Phase space
fig, ax = plt.subplots(figsize=(6,6))
ax.plot(Z, Sigma, lw=1.2)
ax.scatter(Z[in_corner], Sigma[in_corner], c="red", s=10, label="corner dwell")
ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.set_aspect("equal")
ax.set_xlabel("Z"); ax.set_ylabel("Σ")
ax.set_title("Phase Space (Toy-3)")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig); plt.close(fig)

# Tag timeline
fig, ax = plt.subplots(figsize=(8,2.5))
ax.plot(tau, tags, drawstyle="steps-post")
ax.set_yticks([0,1,2])
ax.set_yticklabels(["Local","Coupled","Shared"])
ax.set_xlabel("τ")
ax.set_title("Emergence of Shared Time")
ax.grid(alpha=0.3)
st.pyplot(fig); plt.close(fig)

# =====================================================
# INTERPRETATION
# =====================================================
with st.expander("Interpretation (LOCKED)"):
    st.markdown(
        """
• CSV time is **never** treated as a clock  
• τ emerges only from structural change in Σ  
• Z measures resistance to Σ change  
• Corner dwell = shared time  
• Independent regime is a **valid outcome**

This is a structural test, not a light-curve fit.
"""
    )