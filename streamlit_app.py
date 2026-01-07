import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(
    page_title="Sandy’s Law — Phase Coherence",
    layout="wide"
)

st.title("Sandy’s Law — Phase Coherence")
st.caption("Phase-only instrument • No time • No flux")

# =====================================================
# INPUT
# =====================================================
st.header("1. Paste Phase CSV")

st.write(
    "CSV must contain columns **z** and **sigma**. "
    "`event_id` is optional and ignored."
)

csv_text = st.text_area(
    "Paste CSV here",
    height=240,
    placeholder="event_id,z,sigma\n0,0.62,0.18\n1,0.58,0.21"
)

if not csv_text.strip():
    st.stop()

# =====================================================
# CSV PARSE
# =====================================================
try:
    df = pd.read_csv(StringIO(csv_text))
except Exception as e:
    st.error(f"CSV parse error: {e}")
    st.stop()

df.columns = [c.strip().lower() for c in df.columns]

if "z" not in df.columns or "sigma" not in df.columns:
    st.error("CSV must include columns: z, sigma")
    st.stop()

Z = pd.to_numeric(df["z"], errors="coerce").values
S = pd.to_numeric(df["sigma"], errors="coerce").values

mask = np.isfinite(Z) & np.isfinite(S)
Z, S = Z[mask], S[mask]

N = len(Z)
if N < 6:
    st.error(f"Need at least 6 events. Found {N}.")
    st.stop()

# =====================================================
# CONTROLS (SAFE)
# =====================================================
st.sidebar.header("Controls")

bins = st.sidebar.number_input(
    "Squares per axis",
    min_value=6,
    max_value=100,
    value=24,
    step=2
)

min_shared = st.sidebar.number_input(
    "Min events per square",
    min_value=2,
    max_value=10,
    value=2,
    step=1
)

sweep_max = st.sidebar.number_input(
    "Max N for sweep",
    min_value=4,
    max_value=N,
    value=min(30, N),
    step=1
)

seed = st.sidebar.number_input("Random seed", value=1, step=1)

# =====================================================
# CORE LOGIC
# =====================================================
def coherence(z, s, bins, min_shared):
    zi = np.clip((z * bins).astype(int), 0, bins - 1)
    si = np.clip((s * bins).astype(int), 0, bins - 1)

    occ = np.zeros((bins, bins), dtype=int)
    for a, b in zip(zi, si):
        occ[b, a] += 1

    shared = occ >= min_shared
    count = sum(shared[b, a] for a, b in zip(zi, si))

    return count / len(z), occ, shared

# =====================================================
# FULL DATA RESULT
# =====================================================
C_full, occ_full, shared_full = coherence(Z, S, bins, min_shared)

st.header("2. Full Phase Geometry")

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(Z, S, s=35, alpha=0.85)

ys, xs = np.where(shared_full)
if len(xs):
    ax.scatter(
        (xs + 0.5) / bins,
        (ys + 0.5) / bins,
        s=130,
        marker="s",
        facecolors="none",
        edgecolors="red"
    )

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.set_xlabel("Z (trap)")
ax.set_ylabel("Σ (escape)")
ax.grid(alpha=0.3)
st.pyplot(fig)
plt.close(fig)

st.metric("Coherence C", f"{C_full:.3f}")

# =====================================================
# SWEEP C(N)
# =====================================================
st.header("3. Coherence Sweep C(N)")

rng = np.random.default_rng(seed)

Ns, Cs = [], []
for n in range(4, sweep_max + 1):
    idx = rng.choice(N, size=n, replace=False)
    Cn, _, _ = coherence(Z[idx], S[idx], bins, min_shared)
    Ns.append(n)
    Cs.append(Cn)

Ns = np.array(Ns)
Cs = np.array(Cs)

fig2, ax2 = plt.subplots(figsize=(8, 3))
ax2.plot(Ns, Cs, marker="o")
ax2.set_xlabel("Event count N")
ax2.set_ylabel("Coherence C")
ax2.grid(alpha=0.3)
st.pyplot(fig2)
plt.close(fig2)

# =====================================================
# REGIME
# =====================================================
st.header("4. Regime")

if C_full == 0:
    st.success("Independent regime")
elif C_full < 0.3:
    st.warning("Weak coherence")
elif C_full < 0.7:
    st.warning("Partial macroscopic coherence")
else:
    st.error("Strong macroscopic coherence")

# =====================================================
# NOTES
# =====================================================
with st.expander("What this means"):
    st.write(
        "This instrument measures **phase coherence only**. "
        "No time ordering is used. "
        "Coherence emerges when many events are forced into shared phase squares."
    )