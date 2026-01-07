import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================================
# App Config
# ================================
st.set_page_config(
    page_title="Sandy’s Law — Square → Corner → Quench",
    layout="wide"
)

st.title("Sandy’s Law — Event-Time Phase Mapping")
st.caption("τ-based dynamics | No global clock | Structure-driven time")

# ================================
# Sidebar Controls
# ================================
st.sidebar.header("Controls")

corner_Z = st.sidebar.slider("Corner Z threshold", 0.7, 0.95, 0.85, 0.01)
corner_S = st.sidebar.slider("Corner Σ threshold", 0.7, 0.95, 0.85, 0.01)

show_toy = st.sidebar.checkbox("Overlay Toy 3 trajectory", True)

# ================================
# CSV Input
# ================================
st.subheader("Paste or Upload Event-Time CSV")

csv_text = st.text_area(
    "Paste CSV here (τ, Z, Σ, Tag)",
    height=200,
    placeholder="tau,Z,Sigma,Tag\n0,0.42,0.31,0\n1,0.44,0.33,1\n..."
)

uploaded_file = st.file_uploader("or upload CSV file", type=["csv"])

# ================================
# Load Data
# ================================
df = None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

elif csv_text.strip():
    try:
        df = pd.read_csv(pd.compat.StringIO(csv_text))
    except Exception as e:
        st.error(f"CSV parse error: {e}")

if df is None:
    st.info("Awaiting CSV input.")
    st.stop()

# ================================
# Validate Columns
# ================================
required = {"tau", "Z", "Sigma", "Tag"}
if not required.issubset(df.columns):
    st.error(f"CSV must contain columns: {required}")
    st.stop()

# Sort by event-time
df = df.sort_values("tau").reset_index(drop=True)

# ================================
# Phase-Space Plot
# ================================
fig, ax = plt.subplots(figsize=(6, 6))

ax.plot(
    df["Z"],
    df["Sigma"],
    lw=1.5,
    label="Observed trajectory"
)

# Corner region
ax.axvline(corner_Z, color="red", ls="--", alpha=0.4)
ax.axhline(corner_S, color="red", ls="--", alpha=0.4)

ax.fill_betweenx(
    [corner_S, 1],
    corner_Z,
    1,
    color="red",
    alpha=0.05,
    label="Corner (shared-time)"
)

# ================================
# Toy 3 Overlay
# ================================
if show_toy:
    theta = np.linspace(0, 2*np.pi, 400)
    toy_Z = 0.55 + 0.25 * np.cos(theta)
    toy_S = 0.55 + 0.25 * np.sin(theta)
    ax.plot(toy_Z, toy_S, "--", lw=1.2, label="Toy 3 (template)")

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.set_xlabel("Z (trap strength)")
ax.set_ylabel("Σ (escape)")
ax.set_title("Phase Space (Event-Time Ordered)")
ax.legend()
ax.grid(alpha=0.4)

st.pyplot(fig)
plt.close(fig)

# ================================
# Event-Time Series
# ================================
fig2, ax2 = plt.subplots(figsize=(8, 3))

ax2.plot(df["tau"], df["Z"], label="Z")
ax2.plot(df["tau"], df["Sigma"], label="Σ")

ax2.set_xlabel("τ (event-time)")
ax2.set_ylabel("value")
ax2.set_title("Event-Time Series (τ)")
ax2.legend()
ax2.grid(alpha=0.4)

st.pyplot(fig2)
plt.close(fig2)

# ================================
# Corner Diagnostics (Toy 3)
# ================================
corner_mask = (df["Z"] > corner_Z) & (df["Sigma"] > corner_S)

corner_events = df[corner_mask]

dwell_fraction = len