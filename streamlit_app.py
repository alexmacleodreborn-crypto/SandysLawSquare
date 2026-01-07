import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(
    page_title="Sandy’s Law — Phase Coherence (Toy 3 Overlay)",
    layout="wide",
)

st.title("Sandy’s Law — Phase Coherence (Toy 3 Overlay)")
st.caption("Phase-events only • No time ordering • Shared-time emerges via square crowding")

# =====================================================
# HELPERS
# =====================================================
def _norm_colname(c: str) -> str:
    c = c.strip().lower()
    # allow sigma aliases
    if c in {"Σ", "sig", "sigma", "s"}:
        return "sigma"
    if c in {"z", "trap", "trapstrength", "trap_strength"}:
        return "z"
    if c in {"event", "event_id", "id"}:
        return "event_id"
    return c

def parse_phase_csv(text: str) -> pd.DataFrame:
    if not text or not text.strip():
        raise ValueError("Empty CSV.")
    df = pd.read_csv(StringIO(text))
    df.columns = [_norm_colname(c) for c in df.columns]

    if "z" not in df.columns or "sigma" not in df.columns:
        raise ValueError("CSV must contain columns: z, sigma (event_id optional).")

    # keep only needed cols
    keep = ["event_id", "z", "sigma"] if "event_id" in df.columns else ["z", "sigma"]
    df = df[keep].copy()

    # numeric + drop nans
    df["z"] = pd.to_numeric(df["z"], errors="coerce")
    df["sigma"] = pd.to_numeric(df["sigma"], errors="coerce")
    df = df.dropna(subset=["z", "sigma"])

    # clamp to [0,1] if user expects phase domain
    df["z"] = df["z"].clip(0.0, 1.0)
    df["sigma"] = df["sigma"].clip(0.0, 1.0)

    if "event_id" not in df.columns:
        df.insert(0, "event_id", np.arange(len(df), dtype=int))

    df["event_id"] = pd.to_numeric(df["event_id"], errors="coerce").fillna(np.arange(len(df))).astype(int)
    df = df.sort_values("event_id").reset_index(drop=True)
    return df

def assign_squares(df: pd.DataFrame, bins: int) -> pd.DataFrame:
    # square index in [0..bins-1]
    eps = 1e-12
    zi = np.floor((df["z"].values * (bins - eps))).astype(int)
    si = np.floor((df["sigma"].values * (bins - eps))).astype(int)
    zi = np.clip(zi, 0, bins - 1)
    si = np.clip(si, 0, bins - 1)

    out = df.copy()
    out["z_bin"] = zi
    out["s_bin"] = si
    out["cell"] = out["z_bin"].astype(str) + "_" + out["s_bin"].astype(str)
    return out

def coherence_C_from_occupancy(occ: np.ndarray) -> float:
    """
    Concentration / crowding coherence:
    C = (sum n_i^2 - N) / (N*(N-1))
    - 0 means all events in distinct cells
    - 1 means all events in one cell
    """
    N = int(np.sum(occ))
    if N <= 1:
        return 0.0
    num = float(np.sum(occ.astype(float) ** 2) - N)
    den = float(N * (N - 1))
    return max(0.0, min(1.0, num / den))

def shared_time_fraction(df_binned: pd.DataFrame, min_occ: int) -> float:
    counts = df_binned["cell"].value_counts()
    shared_cells = set(counts[counts >= min_occ].index.tolist())
    if len(df_binned) == 0:
        return 0.0
    return float(np.mean(df_binned["cell"].isin(shared_cells)))

def run_length_stats(states: np.ndarray):
    # run-lengths over event index (not time)
    if len(states) == 0:
        return 0.0, 0
    runs = []
    cur = 1
    for i in range(1, len(states)):
        if states[i] == states[i - 1]:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    return float(np.mean(runs)), int(np.max(runs))

def classify_regime(C: float) -> str:
    # simple bands you can tune
    if C >= 0.75:
        return "Strong macroscopic coherence"
    if C >= 0.50:
        return "Emergent coherence"
    if C >= 0.25:
        return "Weak coherence"
    return "Independent regime (exhaust-free)"

def plot_phase_with_grid(df_binned: pd.DataFrame, bins: int, min_occ: int):
    fig, ax = plt.subplots(figsize=(7, 7))

    # draw grid
    for k in range(1, bins):
        ax.axvline(k / bins, linewidth=0.6, alpha=0.15)
        ax.axhline(k / bins, linewidth=0.6, alpha=0.15)

    # scatter all events
    ax.scatter(df_binned["z"], df_binned["sigma"], s=25, alpha=0.9)

    # highlight shared-time cells
    counts = df_binned["cell"].value_counts()
    shared = counts[counts >= min_occ]
    for cell, n in shared.items():
        zb, sb = cell.split("_")
        zb, sb = int(zb), int(sb)
        x0, y0 = zb / bins, sb / bins
        rect = plt.Rectangle((x0, y0), 1 / bins, 1 / bins, fill=False, linewidth=2)
        ax.add_patch(rect)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Z (trap)")
    ax.set_ylabel("Σ (escape)")
    ax.set_title("Phase Geometry (events; shared squares outlined)")
    ax.grid(True, alpha=0.2)
    ax.set_aspect("equal", adjustable="box")
    st.pyplot(fig)
    plt.close(fig)

def plot_coherence_sweep(df: pd.DataFrame, bins: int):
    # C(N): take first N events by event_id and compute crowding coherence
    Cs = []
    Ns = []
    for N in range(2, len(df) + 1):
        sub = df.iloc[:N].copy()
        subb = assign_squares(sub, bins=bins)
        occ = subb["cell"].value_counts().values
        Cs.append(coherence_C_from_occupancy(occ))
        Ns.append(N)

    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.plot(Ns, Cs, linewidth=1.5)
    ax.set_xlabel("Event count N (by event_id order)")
    ax.set_ylabel("Coherence C")
    ax.set_title("Coherence Sweep C(N)")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

def state_from_event(df_binned: pd.DataFrame, bins: int, crowd_min: int) -> np.ndarray:
    """
    State per event index:
    -1 = under-crowded (independent)
     0 = coherent (in shared-time square)
    +1 = over-crowded (strong crowd; optional)
    """
    counts = df_binned["cell"].value_counts()
    n = df_binned["cell"].map(counts).values

    # map to -1 / 0 / +1
    # -1: n < crowd_min
    #  0: crowd_min <= n < 2*crowd_min
    # +1: n >= 2*crowd_min
    states = np.full(len(df_binned), -1, dtype=int)
    states[(n >= crowd_min) & (n < 2 * crowd_min)] = 0
    states[n >= 2 * crowd_min] = 1
    return states

def plot_state_grid(states: np.ndarray, rows: int = 3):
    """
    Visual square projection of states into a small grid.
    Columns are event-packed (NOT time).
    """
    if len(states) == 0:
        return

    cols = int(np.ceil(len(states) / rows))
    grid = np.full((rows, cols), np.nan)
    for idx, s in enumerate(states):
        r = idx % rows
        c = idx // rows
        grid[r, c] = s

    fig, ax = plt.subplots(figsize=(min(12, 0.6 * cols + 2), 2.5))
    # map -1,0,1 to grayscale positions for display (no custom colors)
    # we just show numeric heatmap; Streamlit theme is fine.
    im = ax.imshow(grid, aspect="auto")
    ax.set_title("State Grid  (-1 | 0 | +1)   (event-packed →)")
    ax.set_xlabel("Square columns (event packed →)")
    ax.set_ylabel("Rows")
    ax.set_yticks(range(rows))
    ax.set_xticks([])
    st.pyplot(fig)
    plt.close(fig)

# =====================================================
# INPUT UI
# =====================================================
st.header("1️⃣ Paste or Upload Phase Event CSV")

st.markdown(
    """
**Expected CSV format (phase-events, not time-series):**

```csv
event_id,z,sigma
0,0.62,0.18
1,0.58,0.21
...