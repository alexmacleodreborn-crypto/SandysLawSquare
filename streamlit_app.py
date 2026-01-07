import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# ==================================================
# Sandy’s Law — Square Crowding → Shared Time
# ==================================================
st.set_page_config(page_title="Sandy’s Law — Square Crowding", layout="wide")

st.title("Sandy’s Law — Square Crowding → Shared Time")
st.caption("Macroscopic phase coherence: increase event density → measure shared-phase overlap (exhaust)")

# ==================================================
# Helpers
# ==================================================
def clamp01(x):
    return np.clip(x, 0.0, 1.0)

def moving_average(x, w):
    if w <= 1:
        return x
    w = int(w)
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")

def safe_norm(x):
    x = np.asarray(x, dtype=float)
    mn, mx = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def parse_csv(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.read().decode("utf-8", errors="replace")
    df = pd.read_csv(StringIO(raw))
    df.columns = [c.strip().lower() for c in df.columns]
    # Accept common variants
    if "index" not in df.columns:
        # try first column as index if named differently
        df.rename(columns={df.columns[0]: "index"}, inplace=True)
    if "flux" not in df.columns:
        # try second column as flux if named differently
        if len(df.columns) >= 2:
            df.rename(columns={df.columns[1]: "flux"}, inplace=True)
    if "index" not in df.columns or "flux" not in df.columns:
        raise ValueError("CSV must contain columns: index, flux (case-insensitive).")

    df = df[["index", "flux"]].copy()
    df = df.dropna()
    df["index"] = pd.to_numeric(df["index"], errors="coerce")
    df["flux"] = pd.to_numeric(df["flux"], errors="coerce")
    df = df.dropna().sort_values("index").reset_index(drop=True)
    return df

def synth_flux(n=6000, n_events=20, noise=0.02, seed=7):
    """
    Synthetic 'index, flux' with bursty events.
    Designed to let you test crowding behaviour without external data.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(n, dtype=float)

    # baseline + noise
    flux = 1.0 + rng.normal(0, noise, size=n)

    # inject events
    centers = rng.choice(np.arange(200, n-200), size=n_events, replace=False)
    centers.sort()

    for c in centers:
        amp = rng.uniform(0.08, 0.35)
        width = rng.uniform(15, 80)
        # asymmetric pulse: quick rise, slower decay
        t = np.arange(n) - c
        rise = np.exp(-(t[t < 0] ** 2) / (2 * (0.35 * width) ** 2))
        decay = np.exp(-(t[t >= 0]) / (0.9 * width))

        pulse = np.zeros(n)
        pulse[t < 0] = amp * rise
        pulse[t >= 0] = amp * decay
        flux += pulse

    return pd.DataFrame({"index": idx, "flux": flux})

def find_events(flux, smooth_w=9, thr_q=0.92, min_sep=60):
    """
    Peak-based event picker on smoothed flux.
    Returns list of peak indices (array positions).
    """
    x = np.asarray(flux, dtype=float)
    xs = moving_average(x, smooth_w)
    thr = np.quantile(xs, thr_q)

    # candidate peaks: local maxima above threshold
    cand = []
    for i in range(1, len(xs)-1):
        if xs[i] > thr and xs[i] >= xs[i-1] and xs[i] >= xs[i+1]:
            cand.append(i)

    # enforce minimum separation by greedy selection (highest first)
    cand = np.array(cand, dtype=int)
    if cand.size == 0:
        return []

    heights = xs[cand]
    order = np.argsort(-heights)
    picked = []
    for j in order:
        i = int(cand[j])
        if all(abs(i - p) >= min_sep for p in picked):
            picked.append(i)
    picked.sort()
    return picked

def event_windows(peaks, n, half_window=120):
    windows = []
    for p in peaks:
        a = max(0, p - half_window)
        b = min(n, p + half_window + 1)
        if b - a >= 10:
            windows.append((a, b, p))
    return windows

def window_to_phase_path(flux_w, mode_z="flux", mode_s="abs_dflux"):
    """
    Convert a flux window into a phase path (Z, Σ) in [0,1].
    IMPORTANT: This does NOT interpret index as time.
    Index only orders samples inside the event.
    """
    f = np.asarray(flux_w, dtype=float)

    # Choose Z mapping
    if mode_z == "flux":
        Z = safe_norm(f)
    elif mode_z == "inv_flux":
        Z = 1.0 - safe_norm(f)
    elif mode_z == "baseline_gap":
        # Z high if event is "trapped" near baseline; low if strongly escapes
        base = np.quantile(f, 0.10)
        gap = f - base
        Z = 1.0 - safe_norm(gap)
    else:
        Z = safe_norm(f)

    # Choose Sigma mapping
    df = np.diff(f, prepend=f[0])
    if mode_s == "abs_dflux":
        S = safe_norm(np.abs(df))
    elif mode_s == "pos_dflux":
        S = safe_norm(np.maximum(df, 0))
    elif mode_s == "neg_dflux":
        S = safe_norm(np.maximum(-df, 0))
    elif mode_s == "cumsum_abs":
        S = safe_norm(np.cumsum(np.abs(df)))
    else:
        S = safe_norm(np.abs(df))

    return clamp01(Z), clamp01(S)

def tile_occupancy(paths, bins=24):
    """
    paths: list of (Z, S) arrays
    returns:
      occ: integer grid counts
      shared_mask: grid cells with count>=2
      shared_fraction: fraction of all samples that land in shared cells
    """
    occ = np.zeros((bins, bins), dtype=int)

    # First pass: count visits per cell (across all samples and paths)
    cell_indices_per_sample = []
    for (Z, S) in paths:
        zi = np.clip((Z * bins).astype(int), 0, bins-1)
        si = np.clip((S * bins).astype(int), 0, bins-1)
        cell_indices_per_sample.append((zi, si))
        for a, b in zip(zi, si):
            occ[b, a] += 1  # row=y(S), col=x(Z)

    shared_mask = occ >= 2

    # Second pass: compute fraction of samples in shared cells
    total = 0
    shared = 0
    for (zi, si) in cell_indices_per_sample:
        total += len(zi)
        shared += np.sum(shared_mask[si, zi])
    shared_fraction = (shared / total) if total > 0 else 0.0

    return occ, shared_mask, shared_fraction

def sweep_crowding(all_paths, n_min=2, n_max=40, step=2, bins=24, seed=1):
    """
    Randomly sample N paths from all_paths, compute exhaust(shared_fraction) vs N.
    """
    rng = np.random.default_rng(seed)
    Ns = []
    exhaust = []
    if len(all_paths) < 2:
        return np.array([]), np.array([])
    n_max = min(n_max, len(all_paths))

    for N in range(n_min, n_max+1, step):
        pick = rng.choice(len(all_paths), size=N, replace=False)
        subset = [all_paths[i] for i in pick]
        _, _, shared_fraction = tile_occupancy(subset, bins=bins)
        Ns.append(N)
        exhaust.append(shared_fraction)
    return np.array(Ns), np.array(exhaust)

def knee_score(Ns, ys):
    """
    Simple knee detector: max second derivative magnitude after smoothing.
    Returns (knee_N, score).
    """
    if len(Ns) < 5:
        return None, 0.0
    y = np.array(ys, dtype=float)
    # smooth
    y_s = moving_average(y, 3)
    # second derivative
    d1 = np.diff(y_s)
    d2 = np.diff(d1)
    if len(d2) == 0:
        return None, 0.0
    k = int(np.argmax(np.abs(d2)))
    # map back to N index (k corresponds to Ns[k+2] roughly)
    kneeN = float(Ns[min(k+2, len(Ns)-1)])
    return kneeN, float(np.max(np.abs(d2)))

# ==================================================
# Sidebar controls
# ==================================================
st.sidebar.header("Input")

source = st.sidebar.radio("Data source", ["Upload CSV (index, flux)", "Use synthetic flux (demo)"])

if source == "Upload CSV (index, flux)":
    up = st.sidebar.file_uploader("Upload CSV", type=["csv"])
else:
    up = None

st.sidebar.header("Event detection (macro: events ≠ time)")

smooth_w = st.sidebar.slider("Smoothing window", 1, 51, 9, 2)
thr_q = st.sidebar.slider("Peak threshold quantile", 0.50, 0.995, 0.92, 0.005)
min_sep = st.sidebar.slider("Min separation (samples)", 10, 400, 60, 5)
half_window = st.sidebar.slider("Event half-window (samples)", 30, 400, 120, 10)

st.sidebar.header("Phase mapping")
mode_z = st.sidebar.selectbox("Z mapping", ["flux", "inv_flux", "baseline_gap"], index=0)
mode_s = st.sidebar.selectbox("Σ mapping", ["abs_dflux", "pos_dflux", "neg_dflux", "cumsum_abs"], index=0)

st.sidebar.header("Square tiling")
bins = st.sidebar.slider("Squares per axis (resolution)", 6, 80, 24, 2)

st.sidebar.header("Crowding sweep")
n_min = st.sidebar.slider("Min events (N)", 2, 60, 2, 1)
n_max = st.sidebar.slider("Max events (N)", 2, 120, 40, 1)
n_step = st.sidebar.slider("Step", 1, 20, 2, 1)
seed = st.sidebar.number_input("Random seed", value=1, step=1)

# ==================================================
# Load data
# ==================================================
try:
    if source == "Use synthetic flux (demo)":
        synth_n = st.sidebar.slider("Synthetic length", 2000, 30000, 6000, 500)
        synth_events = st.sidebar.slider("Synthetic events", 3, 150, 25, 1)
        synth_noise = st.sidebar.slider("Noise", 0.0, 0.20, 0.02, 0.01)
        df = synth_flux(n=synth_n, n_events=synth_events, noise=synth_noise, seed=int(seed))
        st.sidebar.download_button(
            "Download synthetic CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="sandyslaw_synthetic_index_flux.csv",
            mime="text/csv",
        )
    else:
        if up is None:
            st.info("Upload a CSV with columns: index, flux (any casing).")
            st.stop()
        df = parse_csv(up)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

idx = df["index"].to_numpy()
flux = df["flux"].to_numpy()

# ==================================================
# Detect events & build paths
# ==================================================
peaks = find_events(flux, smooth_w=smooth_w, thr_q=thr_q, min_sep=min_sep)
wins = event_windows(peaks, len(flux), half_window=half_window)

paths = []
event_rows = []
for (a, b, p) in wins:
    fwin = flux[a:b]
    Z, S = window_to_phase_path(fwin, mode_z=mode_z, mode_s=mode_s)
    paths.append((Z, S))
    event_rows.append({
        "event_id": len(event_rows),
        "start_i": a,
        "peak_i": p,
        "end_i": b,
        "peak_flux": float(flux[p]),
        "window_len": int(b - a),
    })

events_df = pd.DataFrame(event_rows)

cA, cB, cC = st.columns(3)
cA.metric("Samples", f"{len(df):,}")
cB.metric("Detected events", f"{len(paths):,}")
cC.metric("Square bins", f"{bins}×{bins}")

with st.expander("Preview data"):
    st.dataframe(df.head(30), use_container_width=True)

if len(paths) < 2:
    st.warning("Not enough events detected to run crowding. Lower threshold quantile or min separation.")
    st.stop()

with st.expander("Detected events table"):
    st.dataframe(events_df, use_container_width=True)
    st.download_button(
        "Download detected-events CSV",
        data=events_df.to_csv(index=False).encode("utf-8"),
        file_name="sandyslaw_detected_events.csv",
        mime="text/csv",
    )

# ==================================================
# Sweep crowding: exhaust vs N
# ==================================================
Ns, exhaust = sweep_crowding(
    paths,
    n_min=int(n_min),
    n_max=int(n_max),
    step=int(n_step),
    bins=int(bins),
    seed=int(seed),
)

if Ns.size == 0:
    st.warning("Crowding sweep could not run (need >=2 events).")
    st.stop()

kneeN, kneeStrength = knee_score(Ns, exhaust)

# ==================================================
# Plot: Exhaust vs Event Density
# ==================================================
st.subheader("1) Crowding Curve (Exhaust fraction vs event count)")
fig1, ax1 = plt.subplots(figsize=(8, 3.6))
ax1.plot(Ns, exhaust, marker="o", linewidth=1.5)
ax1.set_xlabel("Event count N (density)")
ax1.set_ylabel("Exhaust fraction (shared-phase overlap)")
ax1.set_title("Crowding → Shared Time (Exhaust)")

if kneeN is not None:
    ax1.axvline(kneeN, linestyle="--")
    ax1.text(kneeN, np.max(exhaust)*0.9, f"Knee ~ N={kneeN:.0f}", rotation=90, va="top")

ax1.grid(True, alpha=0.35)
st.pyplot(fig1)
plt.close(fig1)

c1, c2, c3 = st.columns(3)
c1.metric("Exhaust @ max N", f"{exhaust[-1]*100:.2f}%")
c2.metric("Knee N (rough)", "—" if kneeN is None else f"{kneeN:.0f}")
c3.metric("Knee strength", f"{kneeStrength:.4f}")

st.caption(
    "Exhaust fraction here means: fraction of all samples (across all event paths) that land in phase-squares visited by ≥2 samples."
)

# ==================================================
# Pick a specific N and show phase geometry
# ==================================================
st.subheader("2) Phase Geometry (overlay paths + shared squares)")
N_show = st.slider("Show overlay for N events", int(Ns.min()), int(Ns.max()), int(Ns.max()), int(n_step))

rng = np.random.default_rng(int(seed))
pick = rng.choice(len(paths), size=int(N_show), replace=False)
subset = [paths[i] for i in pick]
occ, shared_mask, shared_fraction = tile_occupancy(subset, bins=int(bins))

fig2, ax2 = plt.subplots(figsize=(6, 6))
for (Z, S) in subset:
    ax2.plot(Z, S, linewidth=1.0, alpha=0.9)

# plot shared squares as red dots at cell centers
ys, xs = np.where(shared_mask)
if len(xs) > 0:
    cx = (xs + 0.5) / bins
    cy = (ys + 0.5) / bins
    ax2.scatter(cx, cy, s=18, marker="o", alpha=0.85, label="Shared-phase dwell")

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_aspect("equal")
ax2.set_xlabel("Z (trap strength proxy)")
ax2.set_ylabel("Σ (escape proxy)")
ax2.set_title(f"Phase Space Overlay (N={N_show})")
ax2.grid(True, alpha=0.35)
ax2.legend(loc="upper right")
st.pyplot(fig2)
plt.close(fig2)

st.subheader("3) Diagnostics")
d1, d2, d3 = st.columns(3)
d1.metric("Exhaust fraction (N show)", f"{shared_fraction*100:.2f}%")
d2.metric("Shared squares", f"{int(shared_mask.sum()):,}")
d3.metric("Occupied squares", f"{int((occ > 0).sum()):,}")

# ==================================================
# Interpretation
# ==================================================
with st.expander("What this means (Sandy’s Law framing)"):
    st.markdown(
        """
**Key idea:** we are not treating *time* as a universal axis.  
We treat the CSV *index* as an **ordering label inside each event**, and we measure structure in **phase geometry**.

### Exhaust (shared time) definition
- We tile phase space into squares.
- If multiple event-path samples land in the same square, that square is a **shared phase window**.
- **Exhaust fraction** = fraction of all samples that fall into **shared squares (count ≥ 2)**.

### Why smaller squares + more events is macroscopic
- Smaller squares = tighter phase tolerance.
- More events = greater population density.
- A sharp rise (“knee”) in exhaust vs N indicates the system crosses into a **shared-time regime** (phase coherence / saturation).

### What to look for
- A **nonlinear knee** in exhaust curve:
  - below knee: mostly independent regime (exhaust ~ 0)
  - above knee: shared regime (exhaust rises fast)

This is the “many becomes one” threshold.
"""
    )