import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

st.set_page_config(page_title="Sandy’s Law — TESS Coherence", layout="wide")
st.title("Sandy’s Law — Real TESS → Phase Coherence C(N)")
st.caption("Real data in-app • time used only for sorting • coherence emerges from phase crowding")

# ------------------------------------------------------
# Helpers
# ------------------------------------------------------
def moving_median(x, w):
    if w <= 1:
        return np.asarray(x, dtype=float)
    return pd.Series(x).rolling(int(w), center=True, min_periods=1).median().values

def safe_norm(x):
    x = np.asarray(x, dtype=float)
    mn, mx = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(mn) or not np.isfinite(mx) or (mx - mn) < 1e-12:
        return np.full_like(x, 0.5)
    return (x - mn) / (mx - mn)

def parse_csv_text(csv_text: str) -> pd.DataFrame:
    """
    Accepts:
      - flux-only (single column, with/without header)
      - time,flux
      - index,flux
    Returns df with columns: order, flux (order is just an ordering label).
    """
    raw = csv_text.strip()
    if not raw:
        raise ValueError("Empty CSV")

    # Try header parse
    try:
        df = pd.read_csv(StringIO(raw))
        df.columns = [str(c).strip().lower() for c in df.columns]
        if "flux" in df.columns:
            flux = pd.to_numeric(df["flux"], errors="coerce").to_numpy()
            if "time" in df.columns:
                t = pd.to_numeric(df["time"], errors="coerce").to_numpy()
                m = np.isfinite(t) & np.isfinite(flux)
                t, flux = t[m], flux[m]
                srt = np.argsort(t)  # order only
                flux = flux[srt]
                order = np.arange(len(flux), dtype=int)
                return pd.DataFrame({"order": order, "flux": flux})
            if "index" in df.columns:
                ix = pd.to_numeric(df["index"], errors="coerce").to_numpy()
                m = np.isfinite(ix) & np.isfinite(flux)
                ix, flux = ix[m], flux[m]
                srt = np.argsort(ix)
                flux = flux[srt]
                order = np.arange(len(flux), dtype=int)
                return pd.DataFrame({"order": order, "flux": flux})

            flux = flux[np.isfinite(flux)]
            order = np.arange(len(flux), dtype=int)
            return pd.DataFrame({"order": order, "flux": flux})
    except Exception:
        pass

    # Fallback: headerless
    df0 = pd.read_csv(StringIO(raw), header=None)
    if df0.shape[1] == 1:
        flux = pd.to_numeric(df0.iloc[:, 0], errors="coerce").to_numpy()
        flux = flux[np.isfinite(flux)]
        return pd.DataFrame({"order": np.arange(len(flux), dtype=int), "flux": flux})

    # assume 2nd col is flux
    flux = pd.to_numeric(df0.iloc[:, 1], errors="coerce").to_numpy()
    flux = flux[np.isfinite(flux)]
    return pd.DataFrame({"order": np.arange(len(flux), dtype=int), "flux": flux})

def find_events(flux, smooth_w=9, thr_q=0.92, min_sep=60):
    x = np.asarray(flux, dtype=float)
    xs = moving_median(x, smooth_w)
    thr = np.quantile(xs, thr_q)
    cand = []
    for i in range(1, len(xs)-1):
        if xs[i] > thr and xs[i] >= xs[i-1] and xs[i] >= xs[i+1]:
            cand.append(i)
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
    out = []
    for p in peaks:
        a = max(0, p - half_window)
        b = min(n, p + half_window + 1)
        if (b - a) >= 10:
            out.append((a, b, p))
    return out

def window_to_phase_path(flux_window, mode_z="baseline_gap", mode_s="abs_dflux"):
    f = np.asarray(flux_window, dtype=float)

    # Z: baseline_gap = "trap" high near baseline, low when escaping strongly
    if mode_z == "flux":
        Z = safe_norm(f)
    elif mode_z == "inv_flux":
        Z = 1.0 - safe_norm(f)
    else:
        base = np.quantile(f, 0.10)
        gap = f - base
        Z = 1.0 - safe_norm(gap)

    df = np.diff(f, prepend=f[0])
    if mode_s == "abs_dflux":
        S = safe_norm(np.abs(df))
    elif mode_s == "pos_dflux":
        S = safe_norm(np.maximum(df, 0))
    elif mode_s == "cumsum_abs":
        S = safe_norm(np.cumsum(np.abs(df)))
    else:
        S = safe_norm(np.abs(df))

    return np.clip(Z, 0, 1), np.clip(S, 0, 1)

def tile_occupancy(paths, bins=24, min_shared=2):
    occ = np.zeros((bins, bins), dtype=int)
    idxs = []
    for (Z, S) in paths:
        zi = np.clip((Z * bins).astype(int), 0, bins-1)
        si = np.clip((S * bins).astype(int), 0, bins-1)
        idxs.append((zi, si))
        for a, b in zip(zi, si):
            occ[b, a] += 1
    shared_mask = occ >= int(min_shared)
    total = 0
    shared = 0
    for (zi, si) in idxs:
        total += len(zi)
        shared += np.sum(shared_mask[si, zi])
    C = (shared / total) if total else 0.0
    return occ, shared_mask, C

def sweep_crowding(paths, bins=24, min_shared=2, n_min=2, n_max=40, step=2, seed=1):
    rng = np.random.default_rng(int(seed))
    n_max = min(int(n_max), len(paths))
    Ns, Cs = [], []
    for N in range(int(n_min), n_max+1, int(step)):
        pick = rng.choice(len(paths), size=N, replace=False)
        subset = [paths[i] for i in pick]
        _, _, C = tile_occupancy(subset, bins=bins, min_shared=min_shared)
        Ns.append(N); Cs.append(C)
    return np.array(Ns), np.array(Cs)

def knee_estimate(Ns, Cs):
    if len(Ns) < 6:
        return None
    y = np.array(Cs, dtype=float)
    y_s = np.convolve(y, np.ones(3)/3, mode="same")
    d1 = np.diff(y_s)
    d2 = np.diff(d1)
    if len(d2) == 0:
        return None
    k = int(np.argmax(np.abs(d2)))
    return int(Ns[min(k+2, len(Ns)-1)])

# ------------------------------------------------------
# Data source
# ------------------------------------------------------
st.sidebar.header("Data source")
mode = st.sidebar.radio("Choose", ["Fetch TESS in-app (recommended)", "Paste CSV"])

@st.cache_data(show_spinner=False)
def fetch_tess_lightcurve_serializable(tic_id: int, author: str, cadence: str, max_points: int) -> pd.DataFrame:
    """
    Returns a SERIALIZABLE DataFrame: columns time, flux.
    We cache only DataFrame to avoid UnserializableReturnValueError.  [oai_citation:2‡docs.streamlit.io](https://docs.streamlit.io/develop/concepts/architecture/caching?utm_source=chatgpt.com)
    """
    import lightkurve as lk  # imported inside cached function

    target = f"TIC {tic_id}"
    # search_lightcurve is the supported API; returns a SearchResult table.  [oai_citation:3‡lightkurve.github.io](https://lightkurve.github.io/lightkurve/reference/api/lightkurve.search_lightcurve.html?highlight=search_lightcurve&utm_source=chatgpt.com)
    sr = lk.search_lightcurve(target, mission="TESS", author=None if author == "Any" else author)

    if len(sr) == 0:
        return pd.DataFrame()

    # download first product (usually best/most direct)
    lc = sr[0].download()
    if lc is None:
        return pd.DataFrame()

    # pick flux column if present
    # lightkurve LightCurve has time and flux arrays
    t = np.array(lc.time.value if hasattr(lc.time, "value") else lc.time)
    f = np.array(lc.flux.value if hasattr(lc.flux, "value") else lc.flux)

    m = np.isfinite(t) & np.isfinite(f)
    t, f = t[m], f[m]
    srt = np.argsort(t)
    t, f = t[srt], f[srt]

    if max_points and len(t) > max_points:
        # simple downsample for mobile friendliness
        idx = np.linspace(0, len(t)-1, max_points).astype(int)
        t, f = t[idx], f[idx]

    return pd.DataFrame({"time": t, "flux": f})

if mode == "Fetch TESS in-app (recommended)":
    st.header("1️⃣ Fetch a real TESS light curve (in-app)")
    st.markdown(
        "Enter a **TIC ID** (e.g., from SIMBAD/ExoFOP), and the app will pull a real light curve from MAST via Lightkurve.  [oai_citation:4‡lightkurve.github.io](https://lightkurve.github.io/lightkurve/tutorials/1-getting-started/using-light-curve-file-products.html?highlight=tesslightcurve&utm_source=chatgpt.com)"
    )
    tic_id = st.number_input("TIC ID", min_value=1, value=141914082, step=1)
    author = st.selectbox("Author", ["Any", "SPOC", "QLP"], index=0)
    cadence = st.selectbox("Cadence", ["Any"], index=0)  # reserved
    max_points = st.slider("Max points (downsample)", 500, 20000, 5000, 500)

    with st.spinner("Fetching TESS light curve..."):
        tess_df = fetch_tess_lightcurve_serializable(int(tic_id), author, cadence, int(max_points))

    if tess_df.empty:
        st.error("No light curve returned for that TIC (or fetch failed). Try another TIC ID or set Author=Any.")
        st.stop()

    # Use time only for ordering; then discard it
    flux = tess_df["flux"].to_numpy()

    with st.expander("Preview TESS data"):
        st.dataframe(tess_df.head(30), use_container_width=True)

    # Optional: let you export CSV right from phone
    st.download_button(
        "Download this lightcurve CSV",
        data=tess_df.to_csv(index=False).encode("utf-8"),
        file_name=f"TIC{int(tic_id)}_tess_lightcurve.csv",
        mime="text/csv",
    )

else:
    st.header("1️⃣ Paste CSV")
    csv_text = st.text_area(
        "Paste (flux-only OR time,flux OR index,flux)",
        height=200,
        placeholder="time,flux\n0.0,1234.5\n0.02,1232.1\n..."
    )
    if not csv_text.strip():
        st.info("Paste CSV to begin.")
        st.stop()
    df = parse_csv_text(csv_text)
    flux = df["flux"].to_numpy()

# ------------------------------------------------------
# Quick plot (real data)
# ------------------------------------------------------
st.header("2️⃣ Raw lightcurve preview (for sanity)")
fig0, ax0 = plt.subplots(figsize=(9, 2.6))
ax0.plot(flux, linewidth=1.0)
ax0.set_xlabel("Sample order (label only)")
ax0.set_ylabel("Flux")
ax0.grid(True, alpha=0.3)
st.pyplot(fig0)
plt.close(fig0)

# ------------------------------------------------------
# Controls: event extraction + coherence
# ------------------------------------------------------
st.sidebar.header("Event extraction")
smooth_w = st.sidebar.slider("Smoothing window", 1, 51, 9, 2)
thr_q = st.sidebar.slider("Peak threshold quantile", 0.50, 0.995, 0.92, 0.005)
min_sep = st.sidebar.slider("Min separation (samples)", 10, 500, 60, 5)
half_window = st.sidebar.slider("Event half-window (samples)", 30, 600, 120, 10)

st.sidebar.header("Phase mapping")
mode_z = st.sidebar.selectbox("Z mapping", ["baseline_gap", "flux", "inv_flux"], index=0)
mode_s = st.sidebar.selectbox("Σ mapping", ["abs_dflux", "pos_dflux", "cumsum_abs"], index=0)

st.sidebar.header("Squares (resolution) + coherence")
bins = st.sidebar.slider("Squares per axis", 6, 120, 24, 2)
min_shared = st.sidebar.slider("Min shared visits per square", 2, 8, 2, 1)

st.sidebar.header("Crowding sweep")
n_min = st.sidebar.slider("Min events N", 2, 120, 2, 1)
n_max = st.sidebar.slider("Max events N", 2, 240, 40, 1)
step = st.sidebar.slider("Step", 1, 30, 2, 1)
seed = st.sidebar.number_input("Random seed", value=1, step=1)

# ------------------------------------------------------
# Build events
# ------------------------------------------------------
peaks = find_events(flux, smooth_w=smooth_w, thr_q=thr_q, min_sep=min_sep)
wins = event_windows(peaks, len(flux), half_window=half_window)

paths = []
events = []
for (a, b, p) in wins:
    Z, S = window_to_phase_path(flux[a:b], mode_z=mode_z, mode_s=mode_s)
    paths.append((Z, S))
    events.append({"event_id": len(events), "start": a, "peak": p, "end": b, "peak_flux": float(flux[p])})

events_df = pd.DataFrame(events)

st.header("3️⃣ Detected events")
c1, c2, c3 = st.columns(3)
c1.metric("Detected events", len(paths))
c2.metric("Square grid", f"{bins}×{bins}")
c3.metric("min_shared", min_shared)

with st.expander("Event table"):
    st.dataframe(events_df, use_container_width=True)

if len(paths) < 2:
    st.warning("Not enough events detected. Lower threshold quantile or min separation.")
    st.stop()

# ------------------------------------------------------
# Crowding curve C(N)
# ------------------------------------------------------
st.header("4️⃣ Macroscopic crowding: C(N)")
Ns, Cs = sweep_crowding(paths, bins=bins, min_shared=min_shared, n_min=n_min, n_max=n_max, step=step, seed=seed)
kneeN = knee_estimate(Ns, Cs)

fig1, ax1 = plt.subplots(figsize=(9, 3.2))
ax1.plot(Ns, Cs, marker="o", linewidth=1.4)
ax1.set_xlabel("Event count N (density)")
ax1.set_ylabel("Coherence C (shared-phase fraction)")
ax1.grid(True, alpha=0.35)
if kneeN is not None:
    ax1.axvline(kneeN, linestyle="--")
    ax1.text(kneeN, max(Cs)*0.9, f"Knee ~ N={kneeN}", rotation=90, va="top")
st.pyplot(fig1)
plt.close(fig1)

d1, d2, d3 = st.columns(3)
d1.metric("C @ max N", f"{Cs[-1]:.3f}")
d2.metric("Knee N (rough)", "—" if kneeN is None else str(kneeN))
d3.metric("Events available", str(len(paths)))

# ------------------------------------------------------
# Overlay for chosen N
# ------------------------------------------------------
st.header("5️⃣ Phase overlay (chosen N)")
N_show = st.slider("Overlay N events", int(Ns.min()), int(Ns.max()), int(Ns.max()), int(step))
rng = np.random.default_rng(int(seed))
pick = rng.choice(len(paths), size=int(N_show), replace=False)
subset = [paths[i] for i in pick]

occ, shared_mask, C_show = tile_occupancy(subset, bins=bins, min_shared=min_shared)

fig2, ax2 = plt.subplots(figsize=(6, 6))
for (Z, S) in subset:
    ax2.plot(Z, S, linewidth=1.0, alpha=0.9)

ys, xs = np.where(shared_mask)
if len(xs) > 0:
    cx = (xs + 0.5) / bins
    cy = (ys + 0.5) / bins
    ax2.scatter(cx, cy, s=20, alpha=0.85, label="Shared squares")

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_aspect("equal")
ax2.set_xlabel("Z (trap proxy)")
ax2.set_ylabel("Σ (escape proxy)")
ax2.grid(True, alpha=0.35)
ax2.legend(loc="upper right")
st.pyplot(fig2)
plt.close(fig2)

e1, e2, e3 = st.columns(3)
e1.metric("C (this overlay)", f"{C_show:.3f}")
e2.metric("Shared squares", f"{int(shared_mask.sum()):,}")
e3.metric("Occupied squares", f"{int((occ > 0).sum()):,}")

with st.expander("What counts as 'real' here (locked)"):
    st.markdown(
        """
- We fetch a real TESS light curve from MAST using Lightkurve.  [oai_citation:5‡lightkurve.github.io](https://lightkurve.github.io/lightkurve/tutorials/1-getting-started/using-light-curve-file-products.html?highlight=tesslightcurve&utm_source=chatgpt.com)  
- Any `time` column is used **only to sort samples** and then discarded (order label only).  
- Coherence C is geometric: shared occupancy of phase squares as event density increases.
"""
    )