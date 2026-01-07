import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# ==================================================
# App config
# ==================================================
st.set_page_config(
    page_title="Sandy’s Law — Emergent Time Mapping",
    layout="wide"
)

st.title("Sandy’s Law — Emergence of Shared Time")
st.caption("CSV → local time τ → Z–Σ space → event tagging")

# ==================================================
# Utility functions
# ==================================================
def normalize_01(x):
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.full_like(x, 0.5)
    return (x - lo) / (hi - lo)

def rolling_median(y, win):
    if win <= 1:
        return y
    return pd.Series(y).rolling(win, center=True, min_periods=1).median().to_numpy()

def grad(y, x):
    if len(y) < 3:
        return np.zeros_like(y)
    g = np.gradient(y, x)
    g[~np.isfinite(g)] = 0.0
    return g

def corner_mask(Z, S, th, code):
    hi = th
    lo = 1 - th
    if code == "UR":
        return (Z >= hi) & (S >= hi)
    if code == "UL":
        return (Z <= lo) & (S >= hi)
    if code == "LR":
        return (Z >= hi) & (S <= lo)
    if code == "LL":
        return (Z <= lo) & (S <= lo)
    return np.zeros_like(Z, dtype=bool)

# ==================================================
# Sidebar — Input
# ==================================================
st.sidebar.header("Input")

uploaded = st.sidebar.file_uploader(
    "Upload CSV (or paste below)",
    type=["csv"]
)

st.sidebar.header("Preprocess")
smooth = st.sidebar.slider("Smoothing window", 1, 51, 9, 2)

st.sidebar.header("Local time τ (NOT constant)")
tau_mode = st.sidebar.radio(
    "τ definition",
    [
        "Index clock τ = n",
        "Arc-length clock τ = ∫|dΣ|",
        "Trap-warped τ = ∫dt·(1 + λZ)"
    ],
    index=2
)
lam = st.sidebar.slider("λ (warp strength)", 0.0, 10.0, 2.0, 0.1)

st.sidebar.header("Toy-3 Tagging")
early_frac = st.sidebar.slider("Early fraction", 0.05, 1.0, 0.30, 0.05)
corner_th = st.sidebar.slider("Corner threshold", 0.60, 0.95, 0.85, 0.01)
corners = st.sidebar.multiselect(
    "Corners to track",
    ["UR", "UL", "LR", "LL"],
    default=["UR", "LR"]
)

# ==================================================
# CSV input (upload OR paste)
# ==================================================
st.subheader("CSV Input")

csv_text = st.text_area(
    "Paste CSV here (time, flux)",
    height=180,
    placeholder="time,flux\n0.0,1.00\n0.1,1.01\n..."
)

if uploaded is None and not csv_text.strip():
    st.info("Upload or paste CSV to begin.")
    st.stop()

if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    df = pd.read_csv(StringIO(csv_text))

if df.shape[1] < 2:
    st.error("CSV must contain at least two columns.")
    st.stop()

time_col = st.selectbox("Time column", df.columns, index=0)
flux_col = st.selectbox("Flux column", df.columns, index=1)

t = pd.to_numeric(df[time_col], errors="coerce").to_numpy()
f = pd.to_numeric(df[flux_col], errors="coerce").to_numpy()

mask = np.isfinite(t) & np.isfinite(f)
t, f = t[mask], f[mask]

order = np.argsort(t)
t, f = t[order], f[order]

if len(t) < 10:
    st.error("Not enough data points.")
    st.stop()

# ==================================================
# Build Σ (escape proxy)
# ==================================================
f_s = rolling_median(f, smooth)
Sigma = normalize_01(f_s)

# ==================================================
# Build local time τ
# ==================================================
dSigma_dn = np.abs(np.gradient(Sigma))
dSigma_dn_n = normalize_01(dSigma_dn)
Z_base = 1.0 - dSigma_dn_n

if tau_mode.startswith("Index"):
    tau = np.arange(len(Sigma), dtype=float)

elif tau_mode.startswith("Arc"):
    tau = np.cumsum(np.maximum(np.abs(np.gradient(Sigma)), 1e-6))

else:
    dt = np.gradient(t)
    d_tau = np.maximum(dt, 0) * (1.0 + lam * Z_base)
    tau = np.cumsum(np.maximum(d_tau, 1e-9))

# ==================================================
# Final Z from τ-dynamics
# ==================================================
dSigma_dtau = np.abs(grad(Sigma, tau))
Z = 1.0 - normalize_01(dSigma_dtau)
Z = np.clip(Z, 0, 1)

# ==================================================
# Tagging
# ==================================================
N = len(tau)
N_early = max(10, int(N * early_frac))
Z_e = Z[:N_early]
S_e = Sigma[:N_early]
tau_e = tau[:N_early]

def dwell_frac(Zw, Sw):
    hit = np.zeros_like(Zw, dtype=bool)
    for c in corners:
        hit |= corner_mask(Zw, Sw, corner_th, c)
    return hit.mean()

DWELL_LOW = 0.03
DWELL_HIGH = 0.10

tags = np.zeros(N, dtype=int)
for i in range(N):
    frac = dwell_frac(Z[:i+1], Sigma[:i+1])
    if frac < DWELL_LOW:
        tags[i] = 0
    elif frac < DWELL_HIGH:
        tags[i] = 1
    else:
        tags[i] = 2

shared_idx = np.where(tags == 2)[0]
shared_tau = tau[shared_idx[0]] if len(shared_idx) else None

# ==================================================
# Output
# ==================================================
st.subheader("Diagnostics")

c1, c2, c3 = st.columns(3)
c1.metric("Early dwell fraction", f"{100*dwell_frac(Z_e, S_e):.2f}%")
c2.metric("Shared-time onset τ", "—" if shared_tau is None else f"{shared_tau:.3f}")
c3.metric("Points", f"{N}")

# ==================================================
# Plots
# ==================================================
st.subheader("Plots")

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(t, f, alpha=0.5, label="raw")
ax.plot(t, f_s, lw=1.2, label="smoothed")
ax.set_title("Light Curve (reference)")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(tau, Sigma, label="Σ")
ax.plot(tau, Z, label="Z")
ax.set_xlabel("τ (local time)")
ax.set_title("Derived variables vs local time")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(Z, Sigma)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.set_xlabel("Z")
ax.set_ylabel("Σ")
ax.set_title("Phase Space (Z, Σ)")
ax.grid(alpha=0.3)
st.pyplot(fig)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 2.5))
ax.plot(tau, tags, drawstyle="steps-post")
ax.set_yticks([0,1,2])
ax.set_yticklabels(["Local", "Coupled", "Shared"])
ax.set_xlabel("τ")
ax.set_title("Event Time → Shared Time Tagging")
ax.grid(alpha=0.3)
st.pyplot(fig)
plt.close(fig)

with st.expander("Interpretation"):
    st.markdown(
        """
**Tag 0 (Local):** independent events, no coordination  
**Tag 1 (Coupled):** events interact / influence timing  
**Tag 2 (Shared):** coherent clock emerges (event onset)

This is **time emerging from structure**, not assumed a priori.
"""
    )