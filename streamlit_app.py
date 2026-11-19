"""Sentinel Space Twin — Streamlit Intelligence Platform"""
from __future__ import annotations

import asyncio
import datetime as dt
import logging
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import ImageOps

sys.path.insert(0, str(Path(__file__).parent / "backend"))

import httpx

from app.constellations import CONSTELLATION_DEFS, INDUSTRIES, collection_windows_for_aoi
from app.constants import PREVIEW_HEIGHT, PREVIEW_WIDTH
from app.enhancement import (
    cell_geojson,
    draw_difference_preview,
    fetch_image,
    pick_thumbnail,
    production_indices,
    visibility_matrix,
)
from app.evidence import generate_record
from app.stac_client import search_sentinel2, summarize_feature
from app.vlm import build_vlm_packet

logging.basicConfig(level=logging.WARNING)

# ── Page ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentinel Space Twin | Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ─────────────────────────────────────────────────────────────
# Bloomberg-style: near-black base, amber accent, muted greens/reds.
# No neon, no developer-palette blues.
#
#  BG_BASE      #0d0d0d   almost pure black
#  BG_SURFACE   #141414   card / panel
#  BG_RAISED    #1c1c1c   elevated element
#  BORDER       #262626   subtle separator
#  BORDER_MED   #333333   visible separator
#  TEXT_HI      #f0ede8   warm near-white  (primary data)
#  TEXT_MED     #a09890   warm mid-grey    (labels, secondary)
#  TEXT_LO      #4a4440   dark grey        (section headers, hints)
#  AMBER        #e8a020   Bloomberg amber  (primary accent)
#  AMBER_DIM    #7a5010   dimmed amber     (subtle highlight)
#  GREEN        #3d9e60   muted green      (positive / strong)
#  RED          #b03030   muted red        (negative / alert)
#  BLUE         #4a7ab0   muted blue       (informational)
#  GOLD         #c8902a   darker gold      (borders on accent elements)

st.markdown("""
<style>
/* ── Global ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #0d0d0d;
    color: #f0ede8;
    font-family: "Inter", "Helvetica Neue", Arial, sans-serif;
}
[data-testid="stMain"] { background: #0d0d0d; }
[data-testid="block-container"] { padding-top: 1.5rem; }
header[data-testid="stHeader"] { background: transparent; border-bottom: 1px solid #262626; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0a0a0a;
    border-right: 1px solid #262626;
}
[data-testid="stSidebar"] * { color: #a09890 !important; }
[data-testid="stSidebar"] .stButton > button { color: #e8a020 !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #141414;
    border: 1px solid #262626;
    border-top: 2px solid #e8a020;
    border-radius: 0;
    padding: 14px 18px;
}
[data-testid="stMetricLabel"] > div {
    color: #4a4440 !important;
    font-size: 10px !important;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] > div {
    color: #f0ede8 !important;
    font-size: 26px !important;
    font-weight: 700;
    letter-spacing: 0.01em;
}
[data-testid="stMetricDelta"] > div {
    font-size: 11px !important;
    letter-spacing: 0.04em;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 0;
    background: transparent;
    border-bottom: 1px solid #262626;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    color: #4a4440 !important;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.12em;
    padding: 10px 24px;
    border-radius: 0;
    border-bottom: 2px solid transparent;
    background: transparent;
    text-transform: uppercase;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {
    color: #a09890 !important;
    background: #141414 !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #e8a020 !important;
    border-bottom: 2px solid #e8a020 !important;
    background: transparent !important;
}

/* ── Primary button ── */
[data-testid="baseButton-primary"] {
    background: #1c1500 !important;
    border: 1px solid #e8a020 !important;
    color: #e8a020 !important;
    font-weight: 700;
    font-size: 11px;
    letter-spacing: 0.14em;
    border-radius: 0;
    text-transform: uppercase;
}
[data-testid="baseButton-primary"]:hover {
    background: #2a2000 !important;
}

/* ── Dataframes ── */
[data-testid="stDataFrame"] {
    border: 1px solid #262626;
}
[data-testid="stDataFrame"] th {
    background: #141414 !important;
    color: #4a4440 !important;
    font-size: 10px !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* ── Dividers ── */
hr { border-color: #262626 !important; }

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: #141414;
    border: 1px solid #262626;
    border-radius: 0;
}
[data-testid="stExpanderToggleIcon"] { color: #4a4440 !important; }

/* ── Inputs ── */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background: #141414 !important;
    border: 1px solid #333333 !important;
    color: #f0ede8 !important;
    border-radius: 0 !important;
    font-size: 12px !important;
}
[data-testid="stSelectbox"] > div > div {
    background: #141414 !important;
    border: 1px solid #333333 !important;
    color: #f0ede8 !important;
    border-radius: 0 !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: #e8a020 !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] { border-radius: 0; border-left-width: 3px; }

/* ── Spinner ── */
[data-testid="stSpinner"] > div { border-top-color: #e8a020 !important; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def _badge(label: str, style: str) -> str:
    palette = {
        "strong": ("#3d9e60", "#0a1a0f"),
        "usable": ("#e8a020", "#1c1500"),
        "weak":   ("#c87820", "#1a1000"),
        "poor":   ("#b03030", "#1a0808"),
        "ok":     ("#3d9e60", "#0a1a0f"),
        "warn":   ("#e8a020", "#1c1500"),
        "error":  ("#b03030", "#1a0808"),
        "info":   ("#4a7ab0", "#08101a"),
    }
    fg, bg = palette.get(style, ("#a09890", "#1c1c1c"))
    return (
        f'<span style="background:{bg};color:{fg};border:1px solid {fg}33;'
        f'padding:2px 10px;font-size:10px;font-weight:700;'
        f'letter-spacing:0.1em;text-transform:uppercase">{label}</span>'
    )


def _kv(key: str, val: str) -> str:
    return (
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
        f'border-bottom:1px solid #1c1c1c;padding:6px 0;font-size:12px">'
        f'<span style="color:#4a4440;letter-spacing:0.04em">{key}</span>'
        f'<span style="color:#f0ede8;font-weight:600;font-variant-numeric:tabular-nums">'
        f'{val}</span></div>'
    )


def _sec(label: str) -> str:
    return (
        f'<div style="font-size:9px;color:#4a4440;letter-spacing:0.18em;'
        f'text-transform:uppercase;margin:20px 0 8px;'
        f'border-bottom:1px solid #1c1c1c;padding-bottom:5px">{label}</div>'
    )


def _note(text: str) -> str:
    return (
        f'<div style="font-size:9px;color:#333333;letter-spacing:0.08em;'
        f'text-transform:uppercase;margin-top:12px;line-height:1.6">{text}</div>'
    )


def _plotly_fin() -> dict:
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0a0a0a",
        font=dict(color="#4a4440", size=10, family="Inter, Helvetica Neue, Arial"),
        margin=dict(l=4, r=4, t=8, b=4),
    )


def _safe(val, fmt: str = "", fallback: str = "—") -> str:
    try:
        return format(val, fmt) if val is not None else fallback
    except Exception:
        return fallback


def _interp_elevation(lon: float, lat: float, elev: dict) -> float:
    """Bilinear interpolation of SRTM elevation grid at (lon, lat)."""
    grid = elev.get("grid")
    bbox = elev.get("bbox")
    n    = elev.get("grid_size", 8)
    if not grid or not bbox or n < 2:
        return 0.0
    minx, miny, maxx, maxy = bbox
    cf = (lon - minx) / max(maxx - minx, 1e-9) * (n - 1)
    rf = (lat - miny) / max(maxy - miny, 1e-9) * (n - 1)
    c0 = int(max(0, min(cf, n - 2)))
    r0 = int(max(0, min(rf, n - 2)))
    c1, r1 = c0 + 1, r0 + 1
    fc, fr = cf - c0, rf - r0
    return (
        grid[r0][c0] * (1 - fc) * (1 - fr)
        + grid[r0][c1] * fc       * (1 - fr)
        + grid[r1][c0] * (1 - fc) * fr
        + grid[r1][c1] * fc       * fr
    )


def _geocode_sync(query: str) -> dict | None:
    if not query.strip():
        return None
    try:
        r = httpx.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": 1},
            headers={"User-Agent": "sentinel-space-twin/2.0"},
            timeout=10,
        )
        data = r.json()
        if data:
            bb   = data[0]["boundingbox"]   # [minlat, maxlat, minlon, maxlon]
            name = data[0]["display_name"].split(",")[0]
            lo, la, hi, ha = float(bb[2]), float(bb[0]), float(bb[3]), float(bb[1])
            # Very large bboxes (whole countries) → 0.3° box around centroid
            if (hi - lo) > 2 or (ha - la) > 2:
                cx, cy = (lo + hi) / 2, (la + ha) / 2
                lo, la, hi, ha = cx - 0.3, cy - 0.3, cx + 0.3, cy + 0.3
            return {"name": name, "bbox": [lo, la, hi, ha]}
    except Exception:
        pass
    return None


def _aoi_map_fig(lon0: float, lat0: float, lon1: float, lat1: float,
                 height: int = 420, scenes: list | None = None) -> go.Figure:
    cx   = (lon0 + lon1) / 2
    cy   = (lat0 + lat1) / 2
    span = max(lon1 - lon0, lat1 - lat0, 0.001)
    zoom = float(np.clip(10 - 3 * np.log10(span), 1, 15))

    fig = go.Figure()
    fig.add_trace(go.Scattermap(
        lon=[lon0, lon1, lon1, lon0, lon0],
        lat=[lat0, lat0, lat1, lat1, lat0],
        mode="lines",
        line=dict(width=2, color="#e8a020"),
        fill="toself",
        fillcolor="rgba(232,160,32,0.08)",
        name="AOI",
        hoverinfo="skip",
    ))
    if scenes:
        s_lons  = [(s["bbox"][0] + s["bbox"][2]) / 2 for s in scenes if s.get("bbox")]
        s_lats  = [(s["bbox"][1] + s["bbox"][3]) / 2 for s in scenes if s.get("bbox")]
        s_texts = [
            f'{s.get("id","")[:24]}<br>Cloud: {s.get("cloud_cover",0):.0f}%  '
            f'Q: {s.get("quality_score",0):.1f}'
            for s in scenes if s.get("bbox")
        ]
        if s_lons:
            fig.add_trace(go.Scattermap(
                lon=s_lons, lat=s_lats,
                mode="markers",
                marker=dict(size=8, color="#e8a020", opacity=0.75),
                text=s_texts, hoverinfo="text",
                name="Scenes",
            ))
    _layout = _plotly_fin()
    _layout["margin"] = dict(l=0, r=0, t=0, b=0)
    fig.update_layout(
        **_layout,
        height=height,
        map=dict(
            style="open-street-map",
            center=dict(lat=cy, lon=cx),
            zoom=zoom,
        ),
        showlegend=False,
    )
    return fig


# ── Static data ───────────────────────────────────────────────────────────────
PRESETS: dict[str, dict] = {
    "rotterdam_port":  {"name": "Rotterdam Port",       "bbox": [3.90, 51.82, 4.65, 52.08]},
    "london_gateway":  {"name": "London Gateway Port",  "bbox": [0.35, 51.44, 0.58, 51.55]},
    "heathrow":        {"name": "London Heathrow",      "bbox": [-0.55, 51.42, -0.36, 51.51]},
    "mumbai_port":     {"name": "Mumbai Port",          "bbox": [72.78, 18.88, 72.93, 19.05]},
    "singapore_port":  {"name": "Singapore Port",       "bbox": [103.60, 1.15, 104.05, 1.38]},
}
_PLABELS = ["Custom"] + [v["name"] for v in PRESETS.values()]
_PKEYS   = [None]      + list(PRESETS.keys())

# Chart series colors — amber family + muted contrasts, no neon
CCOLORS = {
    "blacksky_gen3": "#e8a020",
    "blacksky_gen2": "#c87820",
    "pleiades":      "#a09890",
    "iceye":         "#4a7ab0",
}

# ── Session-state init ────────────────────────────────────────────────────────
_SV = "v6"
if st.session_state.get("_sv") != _SV:
    st.session_state.clear()
    st.session_state["_sv"] = _SV

_DEFAULTS: dict = {
    "_px":    0,
    "_aoi":   "Custom AOI",
    "_lon0":  3.90, "_lat0": 51.82,
    "_lon1":  4.65, "_lat1": 52.08,
    "_loc_gen": 0,          # bumped whenever geocoder/preset changes location
    "_geo_error": False,
    "result": None, "run_error": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def _apply_preset() -> None:
    idx = st.session_state["_px"]
    if not isinstance(idx, int) or idx < 0 or idx >= len(_PKEYS):
        st.session_state["_px"] = 0
        return
    key = _PKEYS[idx]
    if key:
        b = PRESETS[key]["bbox"]
        st.session_state.update(
            _lon0=b[0], _lat0=b[1], _lon1=b[2], _lat1=b[3],
            _aoi=PRESETS[key]["name"],
            _loc_gen=st.session_state["_loc_gen"] + 1,
        )
    # Custom selected: keep whatever name + bbox is currently set


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="font-size:11px;font-weight:700;color:#f0ede8;'
        'letter-spacing:0.14em;text-transform:uppercase;margin-bottom:2px">'
        'Sentinel Space Twin</div>'
        '<div style="font-size:9px;color:#4a4440;letter-spacing:0.12em;'
        'text-transform:uppercase;margin-bottom:16px">Intelligence Platform</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Location search ───────────────────────────────────────────────────────
    st.markdown(_sec("Location"), unsafe_allow_html=True)
    _lg = st.session_state["_loc_gen"]

    _gc1, _gc2 = st.columns([3, 1])
    _geo_query = _gc1.text_input(
        "Search", key="_geo_query",
        placeholder="Search any port or city…",
        label_visibility="collapsed",
    )
    _gc2.markdown("<div style='margin-top:5px'></div>", unsafe_allow_html=True)
    if _gc2.button("Find"):
        st.session_state["_geo_error"] = False
        _geo_res = _geocode_sync(_geo_query or "")
        if _geo_res:
            _gb = _geo_res["bbox"]
            st.session_state.update(
                _lon0=_gb[0], _lat0=_gb[1], _lon1=_gb[2], _lat1=_gb[3],
                _aoi=_geo_res["name"], _px=0,
                _loc_gen=_lg + 1,
            )
            st.rerun()
        else:
            st.session_state["_geo_error"] = True
            st.rerun()

    if st.session_state["_geo_error"]:
        st.warning("Location not found — try a different name.")

    # ── Quick-select presets ───────────────────────────────────────────────────
    st.markdown(
        '<div style="font-size:9px;color:#333333;letter-spacing:0.1em;'
        'text-transform:uppercase;margin:10px 0 4px">or choose a preset</div>',
        unsafe_allow_html=True,
    )
    st.selectbox("Preset", range(len(_PLABELS)),
                 format_func=lambda i: _PLABELS[i],
                 key="_px", on_change=_apply_preset,
                 label_visibility="collapsed")

    # ── Active location — generation-keyed so geocoder/preset always wins ─────
    _lg = st.session_state["_loc_gen"]   # re-read after selectbox may have bumped it

    _name_val = st.text_input(
        "Name", key=f"w_aoi_{_lg}",
        value=st.session_state.get("_aoi", ""),
        placeholder="Enter location name",
        label_visibility="collapsed",
    )
    st.session_state["_aoi"] = _name_val or "Custom AOI"

    _cur_bbox = (f'{st.session_state["_lon0"]:.3f}, {st.session_state["_lat0"]:.3f}'
                 f'  —  {st.session_state["_lon1"]:.3f}, {st.session_state["_lat1"]:.3f}')
    st.markdown(
        f'<div style="font-size:9px;color:#4a4440;font-variant-numeric:tabular-nums;'
        f'padding:2px 0 10px">{_cur_bbox}</div>',
        unsafe_allow_html=True,
    )

    # ── Mini map ──────────────────────────────────────────────────────────────
    _mlon0 = float(st.session_state["_lon0"])
    _mlat0 = float(st.session_state["_lat0"])
    _mlon1 = float(st.session_state["_lon1"])
    _mlat1 = float(st.session_state["_lat1"])
    if _mlon0 < _mlon1 and _mlat0 < _mlat1:
        st.plotly_chart(
            _aoi_map_fig(_mlon0, _mlat0, _mlon1, _mlat1, height=180),
            width="stretch", config={"displayModeBar": False},
        )

    with st.expander("Edit Coordinates", expanded=False):
        c1, c2 = st.columns(2)
        _nlon0 = c1.number_input("Min Lon", value=_mlon0, step=0.0001, format="%.4f",
                                  min_value=-180.0, max_value=180.0, key=f"w_lon0_{_lg}")
        _nlat0 = c2.number_input("Min Lat", value=_mlat0, step=0.0001, format="%.4f",
                                  min_value=-90.0,  max_value=90.0,  key=f"w_lat0_{_lg}")
        _nlon1 = c1.number_input("Max Lon", value=_mlon1, step=0.0001, format="%.4f",
                                  min_value=-180.0, max_value=180.0, key=f"w_lon1_{_lg}")
        _nlat1 = c2.number_input("Max Lat", value=_mlat1, step=0.0001, format="%.4f",
                                  min_value=-90.0,  max_value=90.0,  key=f"w_lat1_{_lg}")
        st.session_state.update(_lon0=_nlon0, _lat0=_nlat0, _lon1=_nlon1, _lat1=_nlat1)

    st.markdown(_sec("Scene Parameters"), unsafe_allow_html=True)
    days_back = st.slider("Days Back",   7,  90, 45)
    max_cloud = st.slider("Max Cloud %", 0, 100, 40)
    limit     = st.slider("Scene Limit", 5,  50, 10, step=5)

    st.divider()
    run_btn = st.button("Run Analysis", type="primary")
    st.divider()

    st.markdown(_sec("Constellation Fleet"), unsafe_allow_html=True)
    for _c in CONSTELLATION_DEFS:
        _col = CCOLORS.get(_c["id"], "#a09890")
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;'
            f'padding:5px 0;border-bottom:1px solid #1c1c1c;font-size:11px">'
            f'<span style="color:{_col}">{_c["name"]}</span>'
            f'<span style="color:#333333">{_c["resolution"]}</span></div>',
            unsafe_allow_html=True,
        )

# ── Header ────────────────────────────────────────────────────────────────────
hc1, hc2 = st.columns([5, 1])
with hc1:
    st.markdown(
        '<h2 style="margin:0;font-size:18px;font-weight:700;color:#f0ede8;'
        'letter-spacing:0.08em;text-transform:uppercase">Sentinel Space Twin</h2>'
        '<div style="font-size:10px;color:#4a4440;letter-spacing:0.14em;'
        'text-transform:uppercase;margin-top:3px">'
        'Satellite Intelligence &nbsp;·&nbsp; Sentinel-2 L2A &nbsp;·&nbsp; '
        'Element84 STAC &nbsp;·&nbsp; Chain-of-Custody Evidence'
        '</div>',
        unsafe_allow_html=True,
    )
with hc2:
    st.markdown(
        f'<div style="text-align:right;color:#333333;font-size:10px;'
        f'padding-top:14px;letter-spacing:0.08em;font-variant-numeric:tabular-nums">'
        f'{dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M")} UTC</div>',
        unsafe_allow_html=True,
    )
st.divider()

# ── Analysis runner ───────────────────────────────────────────────────────────
async def _stac_with_retry(bbox, days, cloud, lim, max_attempts: int = 3) -> list:
    """Call search_sentinel2 with exponential backoff on network errors."""
    last_exc: Exception = RuntimeError("unknown")
    for attempt in range(max_attempts):
        try:
            return await search_sentinel2(bbox, days, cloud, lim)
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                await asyncio.sleep(2 ** attempt)   # 1 s, 2 s
    raise last_exc


async def _analyse(bbox: list, name: str, days: int, cloud: float, lim: int) -> dict:
    features      = await _stac_with_retry(bbox, days, cloud, lim)
    relaxed_cloud: float | None = None
    relaxed_days:  int   | None = None

    # Auto-relax 1: raise cloud ceiling to 80 %
    if not features and cloud < 80:
        relaxed_cloud = 80.0
        features = await _stac_with_retry(bbox, days, relaxed_cloud, lim)

    # Auto-relax 2: extend date window to 90 days
    if not features and days < 90:
        relaxed_days = 90
        features = await _stac_with_retry(bbox, relaxed_days,
                                          relaxed_cloud or cloud, lim)

    scenes = [summarize_feature(f, i) for i, f in enumerate(features)]
    scenes.sort(key=lambda s: (s["quality_score"], s.get("datetime") or ""), reverse=True)

    latest   = scenes[0] if scenes else None
    previous = scenes[1] if len(scenes) > 1 else None

    change_img:    object = None
    change_geojson: dict  = {"type": "FeatureCollection", "features": []}
    change_summary = "Need at least two scenes for comparison."
    change_status  = "not_enough_scenes"

    if latest and previous:
        lat_url = pick_thumbnail(latest)
        pre_url = pick_thumbnail(previous)
        if lat_url and pre_url:
            lat_pil = await fetch_image(lat_url)
            pre_pil = await fetch_image(pre_url)
            if lat_pil and pre_pil:
                sz   = (PREVIEW_WIDTH, PREVIEW_HEIGHT)
                a    = ImageOps.autocontrast(lat_pil.resize(sz).convert("RGB"))
                b    = ImageOps.autocontrast(pre_pil.resize(sz).convert("RGB"))
                diff = np.mean(
                    np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32)),
                    axis=2,
                )
                change_geojson = cell_geojson(diff, bbox)
                change_img     = draw_difference_preview(a, diff)
                n              = len(change_geojson["features"])
                change_status  = "ok"
                change_summary = (
                    f"{n} preview-level changed cell{'s' if n != 1 else ''} detected."
                    if n else "No significant preview-level changes detected."
                )

    # SAM segmentation (best-effort — SLIC fallback if SAM unavailable)
    seg_data: dict | None = None
    elevation_data: dict | None = None
    try:
        from app.sam_segmentation import segment_image as _seg_fn
        from app.elevation import fetch_elevation_grid as _fetch_elev
        _lurl = pick_thumbnail(latest) if latest else None
        if _lurl:
            _limg = await fetch_image(_lurl)
            _purl = pick_thumbnail(previous) if previous else None
            _pimg = await fetch_image(_purl) if _purl else None
            if _limg:
                _loop = asyncio.get_running_loop()
                _limg_snap = _limg
                _pimg_snap = _pimg
                seg_data = await _loop.run_in_executor(
                    None, lambda: _seg_fn(_limg_snap, bbox, _pimg_snap)
                )
                elevation_data = await _fetch_elev(bbox)
    except Exception as _exc:
        logging.warning("SAM segmentation skipped: %s", _exc)

    cw       = collection_windows_for_aoi(bbox)
    evidence = generate_record(
        aoi_name=name, bbox=bbox,
        event=f"Scene capture — {len(scenes)} scenes, change: {change_status}",
        scene_id=latest["id"] if latest else None,
    )
    vlm_pkt = build_vlm_packet(
        name, bbox, latest, previous,
        {
            "status":  change_status,
            "summary": change_summary,
            "geojson": change_geojson,
            "method":  "thumbnail_preview_difference" if change_status == "ok" else "none",
        },
    )
    return {
        "scenes": scenes, "latest": latest, "previous": previous,
        "change_img": change_img, "change_geojson": change_geojson,
        "change_summary": change_summary, "change_status": change_status,
        "cw": cw, "evidence": evidence, "vlm_pkt": vlm_pkt,
        "indices": production_indices(), "vis_matrix": visibility_matrix(),
        "relaxed_cloud": relaxed_cloud,
        "relaxed_days":  relaxed_days,
        "seg_data": seg_data, "elevation_data": elevation_data,
    }


if run_btn:
    lon0 = float(st.session_state["_lon0"])
    lat0 = float(st.session_state["_lat0"])
    lon1 = float(st.session_state["_lon1"])
    lat1 = float(st.session_state["_lat1"])
    name = str(st.session_state["_aoi"]) or "Custom AOI"
    if lon0 >= lon1 or lat0 >= lat1:
        st.error("Invalid bounding box — min values must be strictly less than max values.")
    else:
        with st.spinner(f"Querying Sentinel-2 STAC · {name}"):
            try:
                st.session_state["result"]    = asyncio.run(
                    _analyse([lon0, lat0, lon1, lat1], name, days_back, max_cloud, limit)
                )
                st.session_state["run_error"] = None
            except Exception as exc:
                st.session_state["run_error"] = f"{type(exc).__name__}: {exc}"
                st.session_state["result"]    = None

# ── States ────────────────────────────────────────────────────────────────────
if st.session_state["run_error"]:
    _err = st.session_state["run_error"]
    _net_errs = ("ConnectError", "TimeoutException", "RemoteProtocolError",
                 "ConnectTimeout", "ReadTimeout", "NetworkError")
    if any(_err.startswith(t) for t in _net_errs):
        st.error(
            "**Network error — cannot reach Sentinel-2 STAC catalogue** "
            "(earth-search.aws.element84.com).  "
            "Retried 3 times with backoff.  "
            "Check your internet connection and try again.  "
            f"\n\n`{_err}`"
        )
    else:
        st.error(f"Analysis failed: {_err}")

elif st.session_state["result"] is None:
    st.markdown(
        '<div style="text-align:center;padding:120px 0">'
        '<div style="font-size:10px;letter-spacing:0.22em;text-transform:uppercase;'
        'color:#333333">Select an AOI and run analysis</div>'
        '<div style="font-size:9px;color:#1c1c1c;letter-spacing:0.1em;margin-top:10px">'
        'Sentinel-2 L2A · Element84 STAC · Near real-time EO data</div>'
        '</div>',
        unsafe_allow_html=True,
    )

else:
    r      = st.session_state["result"]
    scenes = r["scenes"]
    latest = r["latest"]
    cw     = r["cw"]

    # ── KPI strip ─────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Scenes Found", len(scenes),
              delta="No results — try wider parameters" if not scenes else None,
              delta_color="off" if not scenes else "normal")
    k2.metric("Best Quality",
              _safe(latest["quality_score"] if latest else None, ".1f"),
              delta=latest["quality_label"].upper() if latest else "No scenes")
    k3.metric("Cloud Cover",
              f'{latest["cloud_cover"]:.0f}%'
              if latest and latest.get("cloud_cover") is not None else "—")
    k4.metric("Next Pass",    cw.get("next_window", "—"))
    k5.metric("Change Cells", len(r["change_geojson"].get("features", [])))

    # ── Relaxed-parameters banner ─────────────────────────────────────────────
    if r.get("relaxed_cloud") or r.get("relaxed_days"):
        parts = []
        if r.get("relaxed_cloud"):
            parts.append(f'cloud cover raised to {r["relaxed_cloud"]:.0f}%')
        if r.get("relaxed_days"):
            parts.append(f'date range extended to {r["relaxed_days"]} days')
        st.info(
            f'No scenes found with your original parameters. '
            f'Results shown after auto-retry: {" and ".join(parts)}. '
            f'Adjust the sliders in the sidebar if needed.'
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    t0, t1, t2, t3, t4, t5, t6 = st.tabs([
        "Map View",
        "Scene Intelligence",
        "Change Detection",
        "3D SAM",
        "Collection Schedule",
        "Evidence Record",
        "Industries",
    ])

    # ── Tab 0 — Map View ─────────────────────────────────────────────────────
    with t0:
        try:
            _mlon0 = float(st.session_state["_lon0"])
            _mlat0 = float(st.session_state["_lat0"])
            _mlon1 = float(st.session_state["_lon1"])
            _mlat1 = float(st.session_state["_lat1"])
            mc1, mc2 = st.columns([4, 1])
            with mc1:
                st.markdown(_sec(f"AOI — {st.session_state.get('_aoi','Custom AOI')}"),
                            unsafe_allow_html=True)
            with mc2:
                st.markdown(
                    f'<div style="font-size:9px;color:#4a4440;text-align:right;'
                    f'padding-top:22px;font-variant-numeric:tabular-nums">'
                    f'{_mlon0:.3f}, {_mlat0:.3f} → {_mlon1:.3f}, {_mlat1:.3f}</div>',
                    unsafe_allow_html=True,
                )
            st.plotly_chart(
                _aoi_map_fig(_mlon0, _mlat0, _mlon1, _mlat1,
                             height=520, scenes=scenes),
                width="stretch",
                config={"displayModeBar": True, "scrollZoom": True},
            )
            st.markdown(
                _note(
                    "Amber box = AOI bounding box · "
                    "dots = Sentinel-2 scene centres · "
                    "hover dots for scene ID and cloud cover · "
                    "scroll to zoom · drag to pan"
                ),
                unsafe_allow_html=True,
            )
        except Exception as exc:
            st.error(f"Map View error: {exc}")
            st.code(traceback.format_exc())

    # ── Tab 1 — Scene Intelligence ────────────────────────────────────────────
    with t1:
        try:
            if not latest:
                st.warning(
                    "No scenes found for this location, even after auto-retrying with "
                    "relaxed cloud cover (80%) and extended date range (90 days).  "
                    "**Check:** is the bounding box over land or ocean with Sentinel-2 "
                    "coverage? Very small or ocean-only bboxes may return no results."
                )
            else:
                lc, rc = st.columns([1, 2])

                with lc:
                    st.markdown(_sec("Best Scene"), unsafe_allow_html=True)
                    st.markdown(
                        _badge(latest.get("quality_label", "—"),
                               latest.get("quality_label", "info")),
                        unsafe_allow_html=True,
                    )
                    st.markdown("<div style='margin-top:12px'></div>",
                                unsafe_allow_html=True)
                    cloud_str = (f'{latest["cloud_cover"]:.1f}%'
                                 if latest.get("cloud_cover") is not None else "—")
                    for k, v in [
                        ("Scene ID",    str(latest.get("id", "—"))),
                        ("Captured",    (latest.get("datetime") or "—")[:19].replace("T", " ")),
                        ("Platform",    str(latest.get("platform") or "—")),
                        ("Cloud Cover", cloud_str),
                        ("Quality",     f'{latest["quality_score"]:.1f} / 100'),
                    ]:
                        st.markdown(_kv(k, v), unsafe_allow_html=True)

                    pts = latest.get("quality_parts") or {}
                    if pts:
                        st.markdown(_sec("Score Breakdown"), unsafe_allow_html=True)
                        fig = go.Figure(go.Bar(
                            x=list(pts.values()),
                            y=["Cloud", "Completeness", "Recency"],
                            orientation="h",
                            marker_color=["#e8a020", "#c87820", "#a09890"],
                        ))
                        fig.update_layout(
                            **_plotly_fin(), height=130,
                            xaxis=dict(range=[0, 100], gridcolor="#1c1c1c",
                                       tickfont=dict(size=9), showline=False),
                            yaxis=dict(gridcolor="#1c1c1c", tickfont=dict(size=9),
                                       showline=False),
                        )
                        st.plotly_chart(fig, width="stretch",
                                        config={"displayModeBar": False})

                    thumb = pick_thumbnail(latest)
                    if thumb:
                        st.markdown(_sec("Scene Thumbnail"), unsafe_allow_html=True)
                        try:
                            st.image(thumb)
                        except Exception:
                            st.caption("Thumbnail unavailable.")

                with rc:
                    st.markdown(_sec(f"All Scenes  ({len(scenes)})"),
                                unsafe_allow_html=True)
                    rows = [{
                        "ID":       str(s.get("id", ""))[:30],
                        "Date":     (s.get("datetime") or "")[:10],
                        "Platform": str(s.get("platform") or "—"),
                        "Cloud %":  f'{s["cloud_cover"]:.0f}' if s.get("cloud_cover") is not None else "—",
                        "Quality":  f'{s["quality_score"]:.1f}',
                        "Grade":    s.get("quality_label", "—").upper(),
                    } for s in scenes]
                    st.dataframe(pd.DataFrame(rows), width="stretch",
                                 hide_index=True,
                                 height=min(400, max(60, 40 + len(rows) * 35)))

                    st.markdown(_sec("Production Spectral Indices"),
                                unsafe_allow_html=True)
                    st.dataframe(
                        pd.DataFrame([{
                            "Index":   i["name"],
                            "Formula": i["formula"],
                            "Meaning": i["meaning"],
                        } for i in r["indices"]]),
                        width="stretch", hide_index=True,
                    )

                    st.markdown(_sec("Visibility Classification"),
                                unsafe_allow_html=True)
                    label_map = {
                        "visible":                   ("Visible",          "ok"),
                        "inferable":                 ("Inferable",        "warn"),
                        "not_visible_from_sentinel": ("Not Visible",      "error"),
                        "needs_external_live_api":   ("Requires Live API","info"),
                    }
                    for cls in r["vis_matrix"]:
                        lbl, style = label_map.get(
                            cls["class"], (cls["class"].replace("_", " ").title(), "info")
                        )
                        st.markdown(_badge(lbl, style), unsafe_allow_html=True)
                        st.markdown(
                            '<div style="font-size:11px;color:#4a4440;'
                            'margin:5px 0 12px 2px;line-height:1.6">'
                            + " · ".join(cls.get("items", []))
                            + "</div>",
                            unsafe_allow_html=True,
                        )
        except Exception as exc:
            st.error(f"Scene Intelligence error: {exc}")
            st.code(traceback.format_exc())

    # ── Tab 2 — Change Detection ───────────────────────────────────────────────
    with t2:
        try:
            if r["change_status"] == "not_enough_scenes":
                st.info(
                    "At least two scenes are required for change detection.  "
                    "**Try:** select **Rotterdam Port** preset, set Days Back to 60, "
                    "Max Cloud % to 50, then re-run."
                )
            else:
                cc1, cc2 = st.columns([3, 2])
                with cc1:
                    st.markdown(_sec("Change Preview"), unsafe_allow_html=True)
                    if r.get("change_img") is not None:
                        st.image(r["change_img"])
                        st.markdown(
                            _note("Preview-quality · thumbnail comparison · "
                                  "production requires raw band + SCL mask processing"),
                            unsafe_allow_html=True,
                        )
                    else:
                        st.warning("Change preview image could not be generated.")

                with cc2:
                    st.markdown(_sec("Detection Summary"), unsafe_allow_html=True)
                    status = r["change_status"]
                    st.markdown(
                        _badge(status.replace("_", " ").title(),
                               "ok" if status == "ok" else "warn"),
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div style="color:#a09890;font-size:12px;margin:14px 0;'
                        f'line-height:1.6">{r["change_summary"]}</div>',
                        unsafe_allow_html=True,
                    )
                    cells = r["change_geojson"].get("features", [])
                    mc1, mc2 = st.columns(2)
                    mc1.metric("Changed Cells", len(cells))
                    mc2.metric("Method", "Thumbnail diff")

                    if cells:
                        st.markdown(_sec("Changed Cells"), unsafe_allow_html=True)
                        st.dataframe(
                            pd.DataFrame([{
                                "Score":  f'{c["properties"]["change_score"]:.3f}',
                                "Mean Δ": f'{c["properties"]["mean_difference"]:.3f}',
                                "Hot Px": f'{c["properties"]["hot_pixel_ratio"]:.1%}',
                            } for c in cells[:20]]),
                            width="stretch", hide_index=True,
                        )
        except Exception as exc:
            st.error(f"Change Detection error: {exc}")
            st.code(traceback.format_exc())

    # ── Tab 3 — 3D SAM ────────────────────────────────────────────────────────
    with t3:
        try:
            import pydeck as pdk

            seg = r.get("seg_data")
            elev = r.get("elevation_data")

            if not seg or not seg.get("features"):
                st.info(
                    "3D SAM segmentation is not available for this scene.  "
                    "scikit-image is required; torch + mobile-sam enable higher-quality SAM "
                    "segments (otherwise SLIC superpixels are used as a fallback)."
                )
            else:
                features = seg.get("features", [])
                method = seg.get("method", "unknown")
                seg_count = seg.get("segment_count", 0)
                sam_ok = seg.get("sam_available", False)

                sm1, sm2, sm3 = st.columns(3)
                sm1.metric("Segments", seg_count)
                sm2.metric("Method", method.replace("_", " ").title())
                sm3.metric("SAM Engine", "MobileSAM" if sam_ok else "SLIC Fallback")

                st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

                _blon0 = float(st.session_state["_lon0"])
                _blat0 = float(st.session_state["_lat0"])
                _blon1 = float(st.session_state["_lon1"])
                _blat1 = float(st.session_state["_lat1"])
                center_lon = (_blon0 + _blon1) / 2
                center_lat = (_blat0 + _blat1) / 2

                # Server-side colour mapping avoids deck.gl property-accessor issues
                _LC_RGBA = {
                    "water":       [30,  110, 220, 200],
                    "vegetation":  [30,  175,  65, 200],
                    "bare_ground": [200, 160,  80, 190],
                    "urban":       [180, 180, 190, 200],
                    "cloud_snow":  [220, 230, 255, 160],
                    "unknown":     [140, 140, 160, 170],
                }
                records   = []
                hex_recs  = []   # change-concentration points for HexagonLayer
                for f in features:
                    props = f["properties"]
                    ring  = f["geometry"]["coordinates"][0]
                    lc    = props.get("land_cover", "unknown")
                    pts   = ring[:-1]   # drop repeated closing vertex
                    clon  = sum(pt[0] for pt in pts) / max(len(pts), 1)
                    clat  = sum(pt[1] for pt in pts) / max(len(pts), 1)
                    # terrain base from SRTM grid (real metres), zero if unavailable
                    terrain = _interp_elevation(clon, clat, elev) if elev else 0.0
                    # Encode terrain in the same unit space as elevation_scale=25:
                    #   vis_h × 25 = (terrain/25 + class_height) × 25
                    # → polygons over hills extrude proportionally higher than sea-level ones
                    vis_h   = round(terrain / 25.0 + props.get("height", 10.0), 1)
                    cs      = round(props.get("change_score", 0.0), 3)
                    records.append({
                        "polygon":      ring,
                        "height_vis":   vis_h,
                        "fill_color":   _LC_RGBA.get(lc, [140, 140, 160, 170]),
                        "land_cover":   lc,
                        "change_score": cs,
                        "area_pct":     props.get("area_pct", 0.0),
                        "terrain_m":    round(terrain, 1),
                    })
                    if cs > 0.01:
                        hex_recs.append({"position": [clon, clat], "weight": cs})

                seg_layer = pdk.Layer(
                    "PolygonLayer",
                    data=records,
                    pickable=True,
                    stroked=True,
                    filled=True,
                    extruded=True,
                    get_polygon="polygon",
                    get_elevation="height_vis",
                    elevation_scale=25,         # vis_h × 25 restores dramatic extrusion
                    get_fill_color="fill_color",
                    get_line_color=[40, 40, 60, 160],
                    line_width_min_pixels=1,
                    opacity=0.82,
                )
                layers = [seg_layer]

                # Change-concentration heatmap — only shown when a prior scene exists
                if hex_recs:
                    layers.append(pdk.Layer(
                        "HexagonLayer",
                        data=hex_recs,
                        pickable=False,
                        extruded=True,
                        radius=300,             # metres per hex bin
                        elevation_scale=6,
                        get_position="position",
                        get_elevation_weight="weight",
                        get_color_weight="weight",
                        elevation_aggregation="MEAN",
                        color_aggregation="MEAN",
                        color_range=[
                            [65,  182, 196, 130],
                            [127, 205, 187, 150],
                            [199, 233, 180, 160],
                            [255, 237, 160, 170],
                            [254, 178,  76, 185],
                            [240,  59,  32, 200],
                        ],
                        opacity=0.55,
                    ))

                view = pdk.ViewState(
                    longitude=center_lon,
                    latitude=center_lat,
                    zoom=11,
                    pitch=50,
                    bearing=10,
                )
                tooltip = {
                    "html": (
                        "<b>{land_cover}</b><br/>"
                        "Extrusion: {height_vis} m &nbsp;"
                        "<span style='color:#a09890'>(terrain base: {terrain_m} m)</span><br/>"
                        "Change score: {change_score}<br/>"
                        "Area: {area_pct}%"
                    ),
                    "style": {
                        "backgroundColor": "#141414",
                        "color": "#f0ede8",
                        "fontSize": "12px",
                        "padding": "8px 12px",
                        "border": "1px solid #333333",
                        "borderRadius": "0",
                    },
                }
                st.pydeck_chart(
                    pdk.Deck(
                        layers=layers,
                        initial_view_state=view,
                        tooltip=tooltip,
                        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
                    )
                )
                st.markdown(
                    _note(
                        "SAM land-cover polygons extruded at real metres · "
                        "terrain base from SRTM 30m · "
                        + ("cyan→red hex bins = change-score concentration · " if hex_recs else "")
                        + "hover polygons for land-cover and terrain detail"
                    ),
                    unsafe_allow_html=True,
                )

                st.markdown(_sec("Land Cover Breakdown"), unsafe_allow_html=True)
                lc_colors = {
                    "water":       "#1e6edc",
                    "vegetation":  "#1eaf41",
                    "bare_ground": "#c8a050",
                    "urban":       "#969698",
                    "cloud_snow":  "#dce6ff",
                    "unknown":     "#787882",
                }
                lc_counts: dict[str, int] = {}
                lc_area: dict[str, float] = {}
                for f in features:
                    lc = f["properties"]["land_cover"]
                    lc_counts[lc] = lc_counts.get(lc, 0) + 1
                    lc_area[lc] = lc_area.get(lc, 0.0) + f["properties"].get("area_pct", 0.0)

                sorted_lc = sorted(lc_counts.items(), key=lambda x: -x[1])
                lc_cols = st.columns(min(len(sorted_lc), 6))
                for idx, (lc, cnt) in enumerate(sorted_lc):
                    col = lc_colors.get(lc, "#888888")
                    with lc_cols[idx % len(lc_cols)]:
                        st.markdown(
                            f'<div style="background:#0a0a0a;border:1px solid #1c1c1c;'
                            f'border-top:3px solid {col};padding:12px;text-align:center;'
                            f'margin-bottom:8px">'
                            f'<div style="font-size:9px;color:#4a4440;text-transform:uppercase;'
                            f'letter-spacing:0.12em">{lc.replace("_"," ")}</div>'
                            f'<div style="font-size:20px;font-weight:700;color:#f0ede8;'
                            f'margin:4px 0 2px">{cnt}</div>'
                            f'<div style="font-size:9px;color:#a09890">'
                            f'{lc_area.get(lc, 0):.1f}% area</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                if elev:
                    st.markdown(_sec("Terrain Elevation (SRTM 30m)"),
                                unsafe_allow_html=True)
                    ec1, ec2, ec3, ec4 = st.columns(4)
                    ec1.metric("Min", f'{elev.get("min_elevation", 0):.0f} m')
                    ec2.metric("Max", f'{elev.get("max_elevation", 0):.0f} m')
                    ec3.metric("Range", f'{elev.get("elevation_range", 0):.0f} m')
                    ec4.metric("Source", str(elev.get("source", "SRTM")).upper())

        except Exception as exc:
            st.error(f"3D SAM error: {exc}")
            st.code(traceback.format_exc())

    # ── Tab 4 — Collection Schedule ───────────────────────────────────────────
    with t4:
        try:
            sc1, sc2 = st.columns([1, 3])
            with sc1:
                st.markdown(_sec("Summary"), unsafe_allow_html=True)
                for k, v in [
                    ("Next Window",     cw.get("next_window", "—")),
                    ("24 h Total",      str(cw.get("total_24h", "—"))),
                    ("Best Resolution", cw.get("best_resolution", "—")),
                ]:
                    st.markdown(_kv(k, v), unsafe_allow_html=True)
                st.markdown(
                    _note("Illustrative schedule.<br>Real windows depend on orbit, "
                          "weather &amp; tasking priority."),
                    unsafe_allow_html=True,
                )

            with sc2:
                windows = cw.get("windows", [])
                if windows:
                    st.markdown(_sec("24-Hour Collection Timeline"),
                                unsafe_allow_html=True)
                    fig = go.Figure()
                    for cid, color in CCOLORS.items():
                        pts = [w for w in windows if w["constellation_id"] == cid]
                        if pts:
                            fig.add_trace(go.Scatter(
                                x=[p["hour_of_day"] for p in pts],
                                y=[pts[0]["constellation_name"]] * len(pts),
                                mode="markers",
                                name=pts[0]["constellation_name"],
                                marker=dict(size=7, color=color, symbol="circle",
                                            opacity=0.85),
                            ))
                    fig.update_layout(
                        **_plotly_fin(), height=220,
                        xaxis=dict(
                            title=dict(text="Hour (UTC)", font=dict(size=9)),
                            gridcolor="#1c1c1c", range=[0, 24], dtick=4,
                            tickfont=dict(size=9), showline=False,
                        ),
                        yaxis=dict(gridcolor="#1c1c1c", tickfont=dict(size=9),
                                   showline=False),
                        legend=dict(bgcolor="rgba(0,0,0,0)",
                                    font=dict(size=10, color="#a09890")),
                        showlegend=True,
                    )
                    st.plotly_chart(fig, width="stretch",
                                    config={"displayModeBar": False})

                    st.markdown(_sec("Next 20 Windows"), unsafe_allow_html=True)
                    st.dataframe(
                        pd.DataFrame([{
                            "Constellation": w["constellation_name"],
                            "UTC":           w["time_utc"],
                            "Mins Away":     w["minutes_from_now"],
                            "Resolution":    w["resolution"],
                            "Type":          w["type"].upper(),
                        } for w in windows[:20]]),
                        width="stretch", hide_index=True,
                    )
        except Exception as exc:
            st.error(f"Collection Schedule error: {exc}")
            st.code(traceback.format_exc())

    # ── Tab 5 — Evidence Record ───────────────────────────────────────────────
    with t5:
        try:
            ev = r.get("evidence", {})
            ec1, ec2 = st.columns(2)

            with ec1:
                st.markdown(_sec("Chain of Custody"), unsafe_allow_html=True)
                for k, v in [
                    ("Record ID",     str(ev.get("id", "—"))),
                    ("AOI Code",      str(ev.get("location", "—"))),
                    ("AOI Name",      str(ev.get("aoi_name", "—"))),
                    ("Captured UTC",  str(ev.get("captured", "—"))[:19]),
                    ("Sensor",        str(ev.get("sensor", "—"))),
                    ("Schema",        str(ev.get("schema_version", "—"))),
                    ("Custody",       str(ev.get("chain_of_custody", "—")).upper()),
                    ("Event",         str(ev.get("event", "—"))),
                ]:
                    st.markdown(_kv(k, v), unsafe_allow_html=True)

            with ec2:
                st.markdown(_sec("SHA-256 Integrity Hash"), unsafe_allow_html=True)
                full_hash = ev.get("integrity_full", "—")
                st.markdown(
                    f'<div style="font-family:\'Courier New\',monospace;font-size:11px;'
                    f'color:#a09890;word-break:break-all;background:#0a0a0a;'
                    f'padding:14px;border:1px solid #262626;line-height:1.8">'
                    f'{full_hash}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div style="font-size:10px;color:#333333;margin-top:8px">'
                    f'{ev.get("integrity", "")}</div>',
                    unsafe_allow_html=True,
                )
                if ev.get("scene_id"):
                    st.markdown(_sec("Scene Reference"), unsafe_allow_html=True)
                    st.markdown(_kv("Scene ID", str(ev["scene_id"])[:40]),
                                unsafe_allow_html=True)
                    st.markdown(_kv("BBox", str(ev.get("bbox", "—"))),
                                unsafe_allow_html=True)

            with st.expander("VLM Analysis Packet"):
                st.json(r.get("vlm_pkt", {}))
        except Exception as exc:
            st.error(f"Evidence Record error: {exc}")
            st.code(traceback.format_exc())

    # ── Tab 6 — Industries ────────────────────────────────────────────────────
    with t6:
        try:
            st.markdown(
                '<div style="font-size:9px;color:#333333;letter-spacing:0.14em;'
                'text-transform:uppercase;margin-bottom:20px">'
                'Monitored sectors — illustrative AOI catalogue</div>',
                unsafe_allow_html=True,
            )
            cols = st.columns(3)
            for i, ind in enumerate(INDUSTRIES):
                with cols[i % 3]:
                    st.markdown(
                        f'<div style="background:#0a0a0a;border:1px solid #1c1c1c;'
                        f'border-top:2px solid #e8a020;padding:16px;margin-bottom:14px">'
                        f'<div style="font-size:9px;color:#4a4440;text-transform:uppercase;'
                        f'letter-spacing:0.12em;margin-bottom:4px">{ind.get("aoi","")}</div>'
                        f'<div style="font-size:13px;font-weight:600;color:#f0ede8;'
                        f'margin-bottom:4px">{ind.get("category","")}</div>'
                        f'<div style="font-size:11px;color:#a09890;margin-bottom:6px">'
                        f'{ind.get("sector","")}</div>'
                        f'<div style="font-size:11px;color:#4a4440;line-height:1.5">'
                        f'{ind.get("description","")}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
        except Exception as exc:
            st.error(f"Industries error: {exc}")
            st.code(traceback.format_exc())
