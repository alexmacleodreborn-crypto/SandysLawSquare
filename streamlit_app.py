# square_projection_app.py
# Streamlit app: Sandy's Law Square Projection (Independent Regime, Exhaust-Free)
#
# CSV expected (minimum):
#   time,value
# Example:
#   0, 1.2
#   1, 1.25
#
# Optional columns:
#   id   (event id)
#
# States output:
#   -1  = "collapse/lock" (negative regime)
#    0  = "coherent/hold" (stable)
#   +1  = "escape/drive"  (positive regime)

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# -----------------------------
# Helpers
# -----------------------------
def robust_z(x: np.ndarray) -> np.ndarray:
    """Robust z-score using median/MAD."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad == 0 or np.isnan(mad):
        return (x - med) * 0.0
    return 0.6745 * (x - med) / mad


def rolling_slope(t: np.ndarray, y: np.ndarray, w: int) -> np.ndarray:
    """Rolling linear slope (least squares) over window size w."""
    n = len(y)
    out = np.full(n, np.nan, dtype=float)
    if w < 3:
        return out

    for i in range(w - 1, n):
        tt = t[i - w + 1 : i + 1]
        yy = y[i - w + 1 : i + 1]
        # handle NaNs
        m = np.isfinite(tt) & np.isfinite(yy)
        if m.sum() < 3:
            continue
        tt2 = tt[m]
        yy2 = yy[m]
        tt2 = tt2 - tt2.mean()
        denom = np.sum(tt2 ** 2)
        if denom <= 0:
            continue
        out[i] = np.sum(tt2 * (yy2 - yy2.mean())) / denom
    return out


def rolling_var(y: np.ndarray, w: int) -> np.ndarray:
    n = len(y)
    out = np.full(n, np.nan, dtype=float)
    if w < 2:
        return out
    for i in range(w - 1, n):
        yy = y[i - w + 1 : i + 1]
        m = np.isfinite(yy)
        if m.sum() < 2:
            continue
        out[i] = np.var(yy[m])
    return out


def assign_state(
    slope_z: np.ndarray,
    var_z: np.ndarray,
    slope_thr: float,
    var_thr: float,
) -> np.ndarray:
    """
    Independent regime (exhaust-free):
      +1 if slope strongly positive and not chaotic
      -1 if slope strongly negative and not chaotic
       0 otherwise (coherent/hold)
    Chaos gate: if var_z is too high, force 0 (system is turbulent/noisy)
    """
    state = np.zeros_like(slope_z, dtype=int)

    # chaos gate
    chaos = np.isfinite(var_z) & (np.abs(var_z) >= var_thr)

    pos = np.isfinite(slope_z) & (slope_z >= slope_thr) & (~chaos)
    neg = np.isfinite(slope_z) & (slope_z <= -slope_thr) & (~chaos)

    state[pos] = 1
    state[neg] = -1
    state[~np.isfinite(slope_z)] = 0
    return state


def run_lengths(arr: np.ndarray):
    """Return list of (start_idx, end_idx, value, length) for runs."""
    runs = []
    if len(arr) == 0:
        return runs
    start = 0
    cur = arr[0]
    for i in range(1, len(arr)):
        if arr[i] != cur:
            runs.append((start, i - 1, cur, i - start))
            start = i
            cur = arr[i]
    runs.append((start, len(arr) - 1, cur, len(arr) - start))
    return runs


def build_square_grid(states: np.ndarray, cols: int) -> np.ndarray:
    """Pack states into a grid with given number of columns."""
    n = len(states)
    rows = int(np.ceil(n / cols))
    grid = np.full((rows, cols), np.nan)
    for i in range(n):
        r = i // cols
        c = i % cols
        grid[r, c] = states[i]
    return grid


def plot_square_grid(grid: np.ndarray, title: str):
    # Map -1,0,1 to numeric palette
    # We'll use a simple discrete colormap.
    from matplotlib.colors import ListedColormap, BoundaryNorm

    cmap = ListedColormap(["#d94b4b", "#cfcfcf", "#3fbf6f"])  # red, grey, green
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(grid, cmap=cmap, norm=norm, interpolation="nearest", aspect="equal")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Square columns (time packed →)")
    ax.set_ylabel("Rows")
    plt.tight_layout()
    return fig


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Square Projection (Sandy’s Law)", layout="wide")
st.title("Square Projection — Independent Regime (Exhaust-Free)")
st.caption("Paste or upload time-series data → compute states (-1/0/+1) → square projection + persistence/alternation/clusters.")

with st.expander("CSV format (copy/paste template)", expanded=False):
    st.code(
        "time,value\n"
        "0,1.00\n"
        "1,1.02\n"
        "2,1.01\n"
        "3,1.05\n"
        "4,1.08\n"
        "5,1.06\n"
        "6,1.10\n"
        "7,1.12\n"
        "8,1.09\n"
        "9,1.15\n"
        "10,1.18\n"
        "11,1.20\n"
    )

left, right = st.columns([1.05, 0.95])

with left:
    st.subheader("1) Load data")
    upload = st.file_uploader("Upload CSV", type=["csv"])
    pasted = st.text_area("…or paste CSV here", height=180, placeholder="time,value\n0,1.0\n1,1.02\n...")

    use_demo = st.toggle("Use demo dataset (12 events)", value=False)

with right:
    st.subheader("2) Settings")
    window = st.slider("Rolling window (events)", 3, 30, 8, 1)
    slope_thr = st.slider("Slope threshold (robust z)", 0.1, 3.0, 0.6, 0.05)
    var_thr = st.slider("Chaos gate: variance threshold (robust z)", 0.1, 5.0, 1.2, 0.05)

    st.markdown("**Square packing**")
    cols = st.slider("Squares per row", 6, 40, 16, 1)

    st.markdown("**Cluster detection**")
    cluster_min_len = st.slider("Min run length to count as cluster", 2, 30, 4, 1)
    cluster_states = st.multiselect("Cluster states to flag", options=[-1, 0, 1], default=[-1, 1])


# -----------------------------
# Load & validate
# -----------------------------
df = None

if use_demo:
    t = np.arange(12)
    y = np.array([1.00, 1.02, 1.01, 1.05, 1.08, 1.06, 1.10, 1.12, 1.09, 1.15, 1.18, 1.20])
    df = pd.DataFrame({"time": t, "value": y})
else:
    raw = None
    if upload is not None:
        raw = upload.read().decode("utf-8", errors="ignore")
    elif pasted.strip():
        raw = pasted

    if raw:
        try:
            df = pd.read_csv(StringIO(raw))
        except Exception as e:
            st.error(f"Could not parse CSV: {e}")
            df = None

if df is None:
    st.info("Load data to begin (upload or paste), or toggle the demo dataset.")
    st.stop()

required = {"time", "value"}
if not required.issubset(set(df.columns)):
    st.error(f"CSV must include columns: {sorted(required)}. Found: {list(df.columns)}")
    st.stop()

df = df.copy()
df = df.sort_values("time").reset_index(drop=True)

# Ensure numeric
df["time"] = pd.to_numeric(df["time"], errors="coerce")
df["value"] = pd.to_numeric(df["value"], errors="coerce")
df = df.dropna(subset=["time", "value"]).reset_index(drop=True)

if len(df) < 12:
    st.warning(f"Only {len(df)} rows loaded. You said you want ~12+ events; this will still run, but visuals are better with more.")

# -----------------------------
# Compute features
# -----------------------------
t = df["time"].to_numpy(dtype=float)
y = df["value"].to_numpy(dtype=float)

slope = rolling_slope(t, y, window)
var = rolling_var(y, window)

slope_z = robust_z(slope)
var_z = robust_z(var)

state = assign_state(slope_z=slope_z, var_z=var_z, slope_thr=slope_thr, var_thr=var_thr)

df["slope"] = slope
df["var"] = var
df["slope_z"] = slope_z
df["var_z"] = var_z
df["state"] = state

# -----------------------------
# Metrics: persistence, alternation, clusters
# -----------------------------
runs = run_lengths(state)

# Persistence: average run length (non-NaN)
run_lens = [r[3] for r in runs]
avg_persist = float(np.mean(run_lens)) if run_lens else 0.0
max_persist = int(np.max(run_lens)) if run_lens else 0

# Alternation: number of state changes per event
changes = int(np.sum(state[1:] != state[:-1])) if len(state) > 1 else 0
alt_rate = float(changes / max(len(state) - 1, 1))

# Cluster detection
clusters = []
for (a, b, v, ln) in runs:
    if (v in cluster_states) and (ln >= cluster_min_len):
        clusters.append({"start_idx": a, "end_idx": b, "state": int(v), "length": int(ln)})

clusters_df = pd.DataFrame(clusters)

# -----------------------------
# Visuals
# -----------------------------
grid = build_square_grid(state, cols=cols)

topA, topB = st.columns([0.55, 0.45])

with topA:
    st.subheader("Square Projection")
    fig = plot_square_grid(grid, title="State Grid  (-1 red | 0 grey | +1 green)")
    st.pyplot(fig, clear_figure=True)

with topB:
    st.subheader("Key scores")
    st.metric("Events", len(df))
    st.metric("Avg persistence (run length)", f"{avg_persist:.2f}")
    st.metric("Max persistence (run length)", f"{max_persist}")
    st.metric("Alternation rate", f"{alt_rate:.3f}")
    st.metric("State changes", f"{changes}")

    st.markdown("---")
    st.write("State counts:")
    counts = pd.Series(state).value_counts().sort_index()
    st.dataframe(counts.rename("count").to_frame(), use_container_width=True)

mid1, mid2 = st.columns([0.6, 0.4])

with mid1:
    st.subheader("Processed table")
    st.dataframe(df, use_container_width=True, height=300)

with mid2:
    st.subheader("Clusters (runs)")
    if clusters_df.empty:
        st.write("No clusters found with current settings.")
    else:
        st.dataframe(clusters_df, use_container_width=True, height=300)

    # quick legend
    st.markdown("**Interpretation (recommended):**")
    st.write("- **+1** = escape/drive (positive slope, not chaotic)")
    st.write("- **0** = coherent/hold (flat, or chaos-gated)")
    st.write("- **-1** = collapse/lock (negative slope, not chaotic)")

# Optional line plot
st.subheader("Signal view")
show_features = st.toggle("Show slope/variance panels", value=True)

fig2, ax = plt.subplots(figsize=(10, 3))
ax.plot(df["time"], df["value"])
ax.set_xlabel("time")
ax.set_ylabel("value")
ax.set_title("Value vs time")
plt.tight_layout()
st.pyplot(fig2, clear_figure=True)

if show_features:
    fig3, ax3 = plt.subplots(figsize=(10, 3))
    ax3.plot(df["time"], df["slope_z"], label="slope_z")
    ax3.axhline(slope_thr, linestyle="--")
    ax3.axhline(-slope_thr, linestyle="--")
    ax3.set_title("Robust slope z-score (thresholded)")
    ax3.set_xlabel("time")
    ax3.set_ylabel("slope_z")
    ax3.legend()
    plt.tight_layout()
    st.pyplot(fig3, clear_figure=True)

    fig4, ax4 = plt.subplots(figsize=(10, 3))
    ax4.plot(df["time"], np.abs(df["var_z"]), label="|var_z|")
    ax4.axhline(var_thr, linestyle="--")
    ax4.set_title("Chaos gate driver (robust |variance z|)")
    ax4.set_xlabel("time")
    ax4.set_ylabel("|var_z|")
    ax4.legend()
    plt.tight_layout()
    st.pyplot(fig4, clear_figure=True)

# -----------------------------
# Download processed CSV
# -----------------------------
st.subheader("Download")
csv_out = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download processed CSV (with states, slope_z, var_z)",
    data=csv_out,
    file_name="square_projection_processed.csv",
    mime="text/csv",
)

st.caption("Tip: if everything keeps turning 0, lower the variance gate threshold OR lower the window size.")