# streamlit_app.py
# =========================================================
# Sandy’s Law — SLED–Square Module (SSM)
# Full drop-in Streamlit module (single-file app)
#
# What it does:
# - Paste/upload event CSV
# - Map events into an NxN Square trap (time optional)
# - Compute Z_square, Sigma_square, G_square, dG/dt, RP alarms
# - Compute cluster stats + weak-gate map
# - Visualize: heatmaps + time series + alarms
#
# Dependencies: streamlit, pandas, numpy, matplotlib
# (No seaborn, no scipy, no networkx required)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Sandy’s Law — SLED–Square Module (SSM)",
    layout="wide",
)

st.title("Sandy’s Law — SLED–Square Module (SSM)")
st.caption("Square Trap • Gate Metrics • Phase-0 Warning (RP) • Weak-Gate Locator")

# =========================================================
# Helpers
# =========================================================

def robust_minmax(series: pd.Series, clip_q=(0.01, 0.99)) -> np.ndarray:
    """Robustly scale series to [0,1] with quantile clipping."""
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x)
    if mask.sum() == 0:
        return np.zeros_like(x, dtype=float)

    lo = np.nanquantile(x[mask], clip_q[0])
    hi = np.nanquantile(x[mask], clip_q[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        # fallback to standard min/max
        lo = np.nanmin(x[mask])
        hi = np.nanmax(x[mask])
        if hi <= lo:
            return np.zeros_like(x, dtype=float)

    x = np.clip(x, lo, hi)
    return (x - lo) / (hi - lo)


def map_to_grid(x01: np.ndarray, y01: np.ndarray, N: int) -> tuple[np.ndarray, np.ndarray]:
    """Map normalized x,y in [0,1] to integer cell coordinates [0..N-1]."""
    x01 = np.clip(x01, 0.0, 1.0)
    y01 = np.clip(y01, 0.0, 1.0)
    u = np.floor(x01 * (N - 1)).astype(int)
    v = np.floor(y01 * (N - 1)).astype(int)
    return u, v


def connected_components(binary_grid: np.ndarray, connectivity: int = 8):
    """Return (num_components, max_component_size) for a 2D boolean grid."""
    assert connectivity in (4, 8)
    H, W = binary_grid.shape
    visited = np.zeros_like(binary_grid, dtype=bool)

    if connectivity == 4:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def inb(r, c):
        return 0 <= r < H and 0 <= c < W

    n_comp = 0
    max_size = 0

    for r in range(H):
        for c in range(W):
            if not binary_grid[r, c] or visited[r, c]:
                continue
            n_comp += 1
            # BFS
            stack = [(r, c)]
            visited[r, c] = True
            size = 0
            while stack:
                rr, cc = stack.pop()
                size += 1
                for dr, dc in nbrs:
                    r2, c2 = rr + dr, cc + dc
                    if inb(r2, c2) and binary_grid[r2, c2] and not visited[r2, c2]:
                        visited[r2, c2] = True
                        stack.append((r2, c2))
            if size > max_size:
                max_size = size

    return n_comp, max_size


def concentration_index(P: np.ndarray) -> float:
    """
    Z_square concentration index based on persistence distribution P (flattened probabilities).
    Returns a value in [0,1] (approximately) where higher => more concentrated (more trapped).
    """
    p = P.flatten()
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    # Normalize to sum=1
    p = p / p.sum()
    # Simpson concentration: sum p^2 in [1/|S|, 1]
    simpson = float(np.sum(p ** 2))
    # Normalize to [0,1] using active support size
    S = p.size
    if S <= 1:
        return 1.0
    # map simpson from [1/S, 1] to [0,1]
    z = (simpson - (1.0 / S)) / (1.0 - (1.0 / S))
    return float(np.clip(z, 0.0, 1.0))


def finite_diff(x: np.ndarray) -> np.ndarray:
    """Simple first difference with same length (prepend 0)."""
    if len(x) == 0:
        return x
    dx = np.zeros_like(x, dtype=float)
    dx[1:] = np.diff(x)
    return dx


def plot_heatmap(grid: np.ndarray, title: str, log_scale: bool = False):
    fig, ax = plt.subplots(figsize=(6, 6))
    data = grid.copy().astype(float)
    if log_scale:
        data = np.log1p(data)
    im = ax.imshow(data, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig, clear_figure=True)


# =========================================================
# Sidebar controls
# =========================================================
with st.sidebar:
    st.header("Square Controls")

    N = st.slider("Grid size (N x N)", min_value=16, max_value=256, value=64, step=16)
    W = st.slider("Persistence window (events)", min_value=50, max_value=5000, value=500, step=50)
    delta = st.slider("Δ step for Σ (events)", min_value=5, max_value=500, value=50, step=5)

    connectivity = st.radio("Cluster connectivity", options=[4, 8], index=1, horizontal=True)

    st.subheader("RP Threshold")
    baseline_len = st.slider("Baseline length (events)", min_value=100, max_value=5000, value=800, step=50)
    k_sigma = st.slider("k · σ for RP trigger", min_value=2.0, max_value=10.0, value=4.0, step=0.5)

    st.subheader("Normalization")
    clip_low = st.slider("Clip low quantile", 0.0, 0.2, 0.01, 0.01)
    clip_high = st.slider("Clip high quantile", 0.8, 1.0, 0.99, 0.01)

    st.subheader("Optional time column")
    use_time = st.checkbox("Use time column for dG/dt (otherwise uses event index)", value=False)

# =========================================================
# Data input
# =========================================================
st.header("1) Input Data")

tab1, tab2 = st.tabs(["Paste CSV", "Upload CSV"])

df = None

with tab1:
    pasted = st.text_area(
        "Paste CSV here",
        height=200,
        placeholder="event_id,x,y\n0,0.62,0.18\n1,0.58,0.21\n..."
    )
    if pasted.strip():
        try:
            df = pd.read_csv(StringIO(pasted))
        except Exception as e:
            st.error(f"Could not parse pasted CSV: {e}")

with tab2:
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up)
        except Exception as e:
            st.error(f"Could not read uploaded CSV: {e}")

if df is None:
    st.info("Provide a CSV via paste or upload to continue.")
    st.stop()

st.success(f"Loaded {len(df):,} rows and {len(df.columns)} columns.")
st.dataframe(df.head(20), use_container_width=True)

# =========================================================
# Column selection
# =========================================================
st.header("2) Choose Embedding Columns")

cols = list(df.columns)
if len(cols) < 2:
    st.error("Need at least 2 numeric columns to embed into Square.")
    st.stop()

c1, c2, c3, c4 = st.columns([1, 1, 1, 1])

with c1:
    x_col = st.selectbox("X feature", options=cols, index=0)
with c2:
    y_col = st.selectbox("Y feature", options=cols, index=min(1, len(cols) - 1))
with c3:
    id_col = st.selectbox("Event ID (optional)", options=["(none)"] + cols, index=0)
with c4:
    t_col = st.selectbox("Time column (optional)", options=["(none)"] + cols, index=0)

# Validate numeric
x01 = robust_minmax(df[x_col], clip_q=(clip_low, clip_high))
y01 = robust_minmax(df[y_col], clip_q=(clip_low, clip_high))

# Time axis
if use_time and t_col != "(none)":
    t_raw = pd.to_numeric(df[t_col], errors="coerce").to_numpy(dtype=float)
    # fallback if bad time
    if np.isfinite(t_raw).sum() < max(10, int(0.1 * len(df))):
        st.warning("Time column has too many non-numeric values; falling back to event index for derivatives.")
        t_axis = np.arange(len(df), dtype=float)
    else:
        # forward-fill missing
        t_series = pd.Series(t_raw).interpolate(limit_direction="both")
        t_axis = t_series.to_numpy(dtype=float)
else:
    t_axis = np.arange(len(df), dtype=float)

u, v = map_to_grid(x01, y01, N)

# =========================================================
# Core computation
# =========================================================
st.header("3) Compute Square Metrics")

n = len(df)
W_eff = min(W, n)
delta_eff = min(delta, max(1, n // 10))
baseline_eff = min(baseline_len, n)

st.caption(
    f"Using: N={N}, W={W_eff}, Δ={delta_eff}, baseline={baseline_eff}, connectivity={connectivity}."
)

# Rolling occupancy window using a ring buffer of last W events
# We'll compute metrics at each step (can be heavy for large n; we’ll sample if needed).
max_points = 5000  # limit computations for very large datasets
step = max(1, int(np.ceil(n / max_points)))

idxs = np.arange(0, n, step)
m = len(idxs)

Z_series = np.zeros(m, dtype=float)
Sigma_series = np.zeros(m, dtype=float)
G_series = np.zeros(m, dtype=float)
clusters_series = np.zeros(m, dtype=int)
max_cluster_series = np.zeros(m, dtype=int)
active_cells_series = np.zeros(m, dtype=int)

# We also keep last computed maps for display
O_last = np.zeros((N, N), dtype=int)
P_last = np.zeros((N, N), dtype=float)
W_last = np.zeros((N, N), dtype=float)

# Maintain rolling window counts in a dict for efficiency
from collections import deque, defaultdict

window = deque()
counts = defaultdict(int)

# Helper to rebuild O map from counts for display
def counts_to_grid(counts_dict):
    grid = np.zeros((N, N), dtype=int)
    for (uu, vv), c in counts_dict.items():
        grid[vv, uu] = c  # note: imshow row=y=v, col=x=u
    return grid

# For Sigma: track unique active cells set size at each checkpoint
active_set = set()

# Also track previous checkpoint active set size
prev_active_size = 0
prev_idx_for_sigma = 0

# Prime through events, update window; compute only at idx checkpoints
checkpoint_set = set(idxs.tolist())

for i in range(n):
    cell = (int(u[i]), int(v[i]))

    # Add
    window.append(cell)
    counts[cell] += 1
    active_set.add(cell)

    # Remove if exceed W_eff
    if len(window) > W_eff:
        old = window.popleft()
        counts[old] -= 1
        if counts[old] <= 0:
            del counts[old]
            # NOTE: active_set is global over all-time; for Sigma we want active-in-window,
            # so we compute active-in-window using counts keys instead.
            # We'll use counts keys for current active support.

    # Compute at checkpoints
    if i in checkpoint_set:
        j = int(np.where(idxs == i)[0][0])

        # Occupancy in window
        O = counts_to_grid(counts)
        O_last = O

        # Persistence map P: frequency in window
        P = O.astype(float) / float(max(1, len(window)))
        P_last = P

        # Active support size |S_t| (in window)
        S_size = int(np.count_nonzero(O))
        active_cells_series[j] = S_size

        # Z_square (concentration of persistence)
        Z = concentration_index(P)
        Z_series[j] = Z

        # Sigma_square: novelty / support growth rate in window over Δ checkpoints
        # Define active support in window as current S_size.
        # Compare to previous checkpoint delta_eff steps back in event index, not time.
        # We implement a simple diff using event indices:
        # Sigma = (S_now - S_prev) / Δ_events
        # We'll update prev at roughly Δ in event index
        if (i - prev_idx_for_sigma) >= delta_eff:
            Sigma = (S_size - prev_active_size) / float(max(1, (i - prev_idx_for_sigma)))
            prev_active_size = S_size
            prev_idx_for_sigma = i
        else:
            # keep last Sigma (smooth)
            Sigma = Sigma_series[j - 1] if j > 0 else 0.0

        Sigma_series[j] = float(Sigma)

        # Gate
        G = (1.0 - Z) * Sigma
        G_series[j] = float(G)

        # Clusters
        binary = (O > 0)
        n_c, s_max = connected_components(binary, connectivity=connectivity)
        clusters_series[j] = int(n_c)
        max_cluster_series[j] = int(s_max)

        # Weak gate map W(u,v) = ΔO * (1-P)
        # Approximate ΔO using local gradient of occupancy relative to mean
        # We build a simple "inflow pressure" proxy: O - rolling mean (global mean in window)
        mean_O = float(O.mean())
        dO_proxy = np.maximum(O.astype(float) - mean_O, 0.0)
        Wmap = dO_proxy * (1.0 - P)
        W_last = Wmap

# Time axis for checkpoint series
t_chk = t_axis[idxs]

# dG/dt (in checkpoint space)
# Use time if provided (t_chk), else event index.
dG = finite_diff(G_series)
dt = finite_diff(t_chk)
dt[dt == 0] = np.nan
dGdt = dG / dt
dGdt[~np.isfinite(dGdt)] = 0.0

# RP threshold based on baseline window (first baseline_eff events -> convert to checkpoints)
# Find checkpoints within baseline event index
baseline_mask = idxs <= min(baseline_eff, n - 1)
base_vals = dGdt[baseline_mask]
mu = float(np.mean(base_vals)) if base_vals.size else 0.0
sig = float(np.std(base_vals)) if base_vals.size else 1e-9
Gamma_crit = mu + k_sigma * sig
RP = (dGdt >= Gamma_crit).astype(int)

# Bundle outputs
out = pd.DataFrame({
    "event_index": idxs,
    "t": t_chk,
    "Z_square": Z_series,
    "Sigma_square": Sigma_series,
    "G_square": G_series,
    "dGdt": dGdt,
    "RP": RP,
    "clusters": clusters_series,
    "max_cluster_size": max_cluster_series,
    "active_cells": active_cells_series,
})

# =========================================================
# Display results
# =========================================================
st.header("4) Results")

cA, cB, cC = st.columns([1, 1, 1])

with cA:
    st.metric("RP threshold Γ_crit", f"{Gamma_crit:.6g}")
with cB:
    st.metric("Latest Z_square", f"{out['Z_square'].iloc[-1]:.4f}")
with cC:
    st.metric("Latest G_square", f"{out['G_square'].iloc[-1]:.6g}")

# Time series charts
left, right = st.columns([1, 1])

with left:
    st.subheader("Gate Metrics")
    st.line_chart(out.set_index("t")[["Z_square", "Sigma_square", "G_square"]], height=260)

    st.subheader("Phase-0 Warning (RP)")
    # Plot dG/dt with threshold + RP marks
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(out["t"], out["dGdt"], label="dG/dt")
    ax.axhline(Gamma_crit, linestyle="--", label="Γ_crit")
    # RP markers
    rp_idx = out["RP"].to_numpy(dtype=bool)
    ax.scatter(out["t"][rp_idx], out["dGdt"][rp_idx], s=15, label="RP")
    ax.set_xlabel("t" if (use_time and t_col != "(none)") else "event index (proxy time)")
    ax.set_ylabel("dG/dt")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

with right:
    st.subheader("Structural Stats")
    st.line_chart(out.set_index("t")[["clusters", "max_cluster_size", "active_cells"]], height=260)

    st.subheader("Output Table (checkpoints)")
    st.dataframe(out.tail(30), use_container_width=True)

# Heatmaps
st.header("5) Square Maps (latest window)")

h1, h2, h3 = st.columns([1, 1, 1])

with h1:
    plot_heatmap(O_last, "Occupancy O (window)", log_scale=True)

with h2:
    plot_heatmap(P_last, "Persistence P (window)", log_scale=False)

with h3:
    plot_heatmap(W_last, "Weak-Gate Map W (window)", log_scale=True)

# Download
st.header("6) Export")
csv_bytes = out.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download checkpoint metrics CSV",
    data=csv_bytes,
    file_name="sled_square_metrics.csv",
    mime="text/csv",
)

st.caption(
    "Notes: If you enable a time column, dG/dt uses that axis. Otherwise it uses event index as proxy-time. "
    "For very large datasets, the module auto-subsamples checkpoints to keep it responsive."
)