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
    layout="wide",
)

st.title("Sandy’s Law — Phase Coherence Instrument")
st.caption("Real phase events • No time • Coherence emergence via square crowding")

# =====================================================
# CSV INPUT
# =====================================================
st.header("1️⃣ Paste Phase Event CSV")

st.markdown(
    """
**Expected CSV format (from Colab / TESS mapping):**

```csv
event_id,z,sigma
0,0.62,0.18
1,0.58,0.21
...