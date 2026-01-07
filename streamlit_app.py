import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(
    page_title="Sandy’s Law — Phase Coherence Sweep",
    layout="wide"
)

st.title("Sandy’s Law — Phase Coherence Instrument")
st.caption("Real TESS → phase events • No time • Coherence emergence via crowding")

# =====================================================
# INPUT
# =====================================================
st.header("1️⃣ Paste Phase Event CSV")

st.markdown(
    """
**CSV format (from Colab):**