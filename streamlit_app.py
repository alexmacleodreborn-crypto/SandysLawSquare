import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightkurve as lk

st.set_page_config(page_title="Sandy’s Law — Square (TESS)", layout="wide")

st.title("Sandy’s Law — Square Phase Space (TESS)")
st.caption("Real light curves → Toy 3 (Corner Dwell → Quench)")

# ============================================================
# PREPROCESSING OPTIONS
# ============================================================

normalize = st.checkbox("Normalize Flux", value=True)

smooth = st.slider(
    "Smoothing window (odd number)",
    min_value=1,
    max_value=101,
    value=15,
    step=2
)
# ------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------
target = st.text_input("TESS Target (TIC / KIC / Name)", "TIC 307210830")
sector = st.number_input(
    "TESS Sector (0 = auto)",
    min_value=0,
    max_value=100,
    value=0,
    step=1
)
# ============================================================
# LOAD TESS LIGHT CURVE (ROBUST)
# ============================================================

df = None

with st.spinner("Loading TESS light curve..."):
    search = lk.search_lightcurve(target, mission="TESS")

    if len(search) == 0:
        st.error("No TESS products found for this target.")
        st.stop()

    lc = search.download()

    if lc is None:
        st.error(
            "TESS search returned results, but no downloadable light curve "
            "is available for this target.\n\n"
            "Try a different target."
        )
        st.stop()

    lc = lc.remove_nans()

    df = pd.DataFrame({
        "time": lc.time.value,
        "flux": lc.flux.value
    })

# ============================================================
# SAFETY CHECK (MANDATORY)
# ============================================================

if df is None or df.empty:
    st.error("Light curve failed to load correctly.")
    st.stop()
# ------------------------------------------------------------
# PREPROCESS
# ------------------------------------------------------------
flux = df["flux"].values
time = df["time"].values

if normalize:
    flux = (flux - np.min(flux)) / (np.max(flux) - np.min(flux))

if smooth > 1:
    flux = np.convolve(flux, np.ones(smooth)/smooth, mode="same")

# ------------------------------------------------------------
# MAP → SANDY’S LAW VARIABLES
# ------------------------------------------------------------
# Z = trap strength (structure)
# Σ = entropy escape (radiative change)

dflux = np.gradient(flux)

Z = 1.0 - flux
Sigma = np.abs(dflux)

# normalize both to [0,1]
Z = (Z - Z.min()) / (Z.max() - Z.min())
Sigma = (Sigma - Sigma.min()) / (Sigma.max() - Sigma.min())

# ------------------------------------------------------------
# TOY 3 DIAGNOSTICS
# ------------------------------------------------------------
corner_mask = (Z > 0.8) & (Sigma < 0.2)
corner_fraction = corner_mask.sum() / len(Z)

# ------------------------------------------------------------
# PLOTS
# ------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Phase Space: Z vs Σ (Toy 3)")
    fig, ax = plt.subplots()
    ax.plot(Z, Sigma, lw=0.8)
    ax.set_xlabel("Z (Trap Strength)")
    ax.set_ylabel("Σ (Entropy Escape)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    st.pyplot(fig)

with col2:
    st.subheader("Time Series")
    fig, ax = plt.subplots()
    ax.plot(Z, label="Z")
    ax.plot(Sigma, label="Σ")
    ax.legend()
    ax.set_xlabel("Time step")
    st.pyplot(fig)

# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------
st.subheader("Corner Dwell Diagnostics")
st.metric("Corner dwell fraction", f"{corner_fraction*100:.2f}%")

# ------------------------------------------------------------
# INTERPRETATION
# ------------------------------------------------------------
st.markdown("""
### Physical Meaning (Toy 3)

• **High Z + Low Σ** → energy trapped, structure dominates  
• **Corner dwell** → system hesitates before release  
• **Quench** → rapid exit from trap after dwell  

This is the **exact same signature** predicted for:
- Early supernova photon escape  
- Shock breakout precursors  
- Magnetic reconnection delays  
- Sandy’s Law collapse → release regime
""")