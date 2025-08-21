# -*- coding: utf-8 -*-
r"""
Fantasy Draft Console — Streamlit UI (auto data folder; live CSV + team_manager selector)

- Loads projections & ADP from ./data next to this script (portable folder).
- Reads *live* draft picks from ./data/liveData/*.csv and:
    • Marks those players as unavailable.
    • Uses the **team_manager** column to choose which franchise is **your** team, then
      shows your picks automatically (no manual buttons).
- Removes "taken/hide/draft" buttons; availability is driven entirely by the live CSV.
- Fortified ADP parsing: always yields ["Player","Pos","ADP"] to avoid engine KeyErrors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, List, Set
from datetime import datetime
import io
import csv

import numpy as np
import pandas as pd
import streamlit as st

from fantasy_draft_engine import (
    LeagueConfig, VarianceModel, InjuryModel,
    load_fp_uploads,            # projections loader from engine
    build_model_df, compute_targets, pick_number
)

# --- Position canonicalization helpers (UI side to mirror engine) ---
import re as _re  # local alias to avoid confusion
def _canon_pos_str_ui(x: str) -> str:
    s = str(x).upper()
    s = _re.sub(r"[^A-Z/]", "", s).replace("/", "")
    if s.startswith("QB"):
        return "QB"
    if s.startswith("RB"):
        return "RB"
    if s.startswith("WR"):
        return "WR"
    if s.startswith("TE"):
        return "TE"
    if s in ("K", "PK"):
        return "K"
    if s in ("DST", "DEF"):
        return "DST"
    return s

def _canon_pos_series_ui(s: pd.Series) -> pd.Series:
    return s.astype(str).map(_canon_pos_str_ui)


# =============================================================================
# Page config
# =============================================================================
st.set_page_config(page_title="Fantasy Draft Console (Live CSV)", layout="centered")
st.title("Draft Console — Live CSV")

# Top-of-page refresh button
if st.button("🔄 Refresh live picks", key="refresh_live_top", use_container_width=True):
    st.cache_data.clear()
    st.rerun()


# =============================================================================
# Paths & discovery
# =============================================================================
def _app_root() -> Path:
    """Folder containing this script (fallback to CWD if needed)."""
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()

def _resolve_data_dir() -> Path:
    """Prefer <app>/data; fallback to CWD/data. Create if missing."""
    candidates = [_app_root() / "data", Path.cwd() / "data"]
    for d in candidates:
        if d.exists() and d.is_dir():
            return d
    d = candidates[0]
    d.mkdir(parents=True, exist_ok=True)
    return d

def _resolve_live_dir() -> Path:
    """<data>/liveData . Create if missing."""
    d = _resolve_data_dir() / "liveData"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _pick_latest(paths: List[Path]) -> Optional[Path]:
    if not paths:
        return None
    try:
        return max(paths, key=lambda p: p.stat().st_mtime)
    except Exception:
        return paths[0]

def _find_by_exact(data_dir: Path, filename: str) -> Optional[Path]:
    p = data_dir / filename
    return p if p.exists() and p.is_file() else None

def _find_by_keywords(data_dir: Path, must_have: List[str], must_not: Optional[List[str]] = None) -> Optional[Path]:
    """Case-insensitive filename match for *.csv in folder."""
    must_not = must_not or []
    hits: List[Path] = []
    for p in data_dir.glob("*.csv"):
        name = p.name.lower()
        if all(k.lower() in name for k in must_have) and all(n.lower() not in name for n in must_not):
            hits.append(p)
    return _pick_latest(hits)

def _discover_files(data_dir: Path) -> Dict[str, Optional[Path]]:
    """
    Projections/ADP files. Keys: qb, flx, k, dst, adp
    """
    exact = {
        "qb":  "FantasyPros_Fantasy_Football_Projections_QB.csv",
        "flx": "FantasyPros_Fantasy_Football_Projections_FLX.csv",
        "k":   "FantasyPros_Fantasy_Football_Projections_K.csv",
        "dst": "FantasyPros_Fantasy_Football_Projections_DST.csv",
        "adp": "FantasyPros_2025_Overall_ADP_Rankings.csv",
    }
    out: Dict[str, Optional[Path]] = {k: _find_by_exact(data_dir, v) for k, v in exact.items()}

    if out["flx"] is None:
        out["flx"] = (_find_by_keywords(data_dir, ["flx"])
                      or _find_by_keywords(data_dir, ["rb", "wr", "te"]))
    if out["qb"] is None:
        out["qb"] = _find_by_keywords(data_dir, ["qb"])
    if out["k"] is None:
        candidates: List[Path] = []
        for p in data_dir.glob("*.csv"):
            name = p.name.lower()
            if "kicker" in name or name.endswith("_k.csv"):
                candidates.append(p)
        out["k"] = _pick_latest(candidates) or _find_by_keywords(data_dir, ["k"])
    if out["dst"] is None:
        out["dst"] = _find_by_keywords(data_dir, ["dst"]) or _find_by_keywords(data_dir, ["def"])
    if out["adp"] is None:
        out["adp"] = (_find_by_keywords(data_dir, ["2025", "adp"])
                      or _find_by_keywords(data_dir, ["overall", "adp"])
                      or _find_by_keywords(data_dir, ["adp"]))
    return out

def _read_bytes(p: Optional[Path]) -> Optional[bytes]:
    if p is None:
        return None
    try:
        return p.read_bytes()
    except Exception:
        return None

def _auto_read_bytes(files: Dict[str, Optional[Path]]) -> Tuple[Optional[bytes], Optional[bytes], Optional[bytes], Optional[bytes], Optional[bytes]]:
    qb_b  = _read_bytes(files.get("qb"))
    flx_b = _read_bytes(files.get("flx"))
    k_b   = _read_bytes(files.get("k"))
    dst_b = _read_bytes(files.get("dst"))
    adp_b = _read_bytes(files.get("adp"))
    return qb_b, flx_b, k_b, dst_b, adp_b

# =============================================================================
# ADP: robust parsing (always returns Player/Pos/ADP)
# =============================================================================
def _robust_parse_adp(adp_b: Optional[bytes]) -> Optional[pd.DataFrame]:
    """
    Best-effort ADP parser that tolerates messy lines and odd headers.
    Guarantees a DataFrame with columns ["Player","Pos","ADP"] (NaN-filled if missing).
    """
    if not adp_b:
        return None

    # Sniff delimiter (BOM-safe)
    buf = io.BytesIO(adp_b)
    sample = buf.read(8192).decode("utf-8", errors="ignore")
    try:
        dialect = csv.Sniffer().sniff(sample)
        delim = dialect.delimiter
    except Exception:
        delim = ","
    buf.seek(0)

    try:
        df = pd.read_csv(
            buf, engine="python", sep=delim,
            encoding="utf-8-sig", on_bad_lines="skip"
        )
    except Exception:
        return None

    if df is None or df.empty:
        return None

    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}

    # Identify likely columns
    name_col = lower.get("player") or lower.get("player name") or lower.get("name") or list(df.columns)[0]
    pos_col  = lower.get("pos") or lower.get("position")
    adp_col  = (lower.get("adp") or lower.get("overall adp") or lower.get("overall_adp")
                or lower.get("avg draft position") or lower.get("average draft position")
                or lower.get("avg. draft position") or lower.get("overall"))

    out = pd.DataFrame({"Player": df[name_col].astype(str).str.strip()})
    if pos_col:
        out["Pos"] = _canon_pos_series_ui(df[pos_col])
    else:
        out["Pos"] = pd.NA
    if adp_col:
        out["ADP"] = pd.to_numeric(df[adp_col], errors="coerce")
    else:
        out["ADP"] = np.nan

    # Drop blanks
    out = out[out["Player"].astype(str).str.strip().ne("")]
    if out.empty:
        return None

    # Ensure the needed columns exist
    for needed in ("Player", "Pos", "ADP"):
        if needed not in out.columns:
            out[needed] = pd.NA
    return out

# =============================================================================
# Live picks parsing & mapping to engine PIDs
# =============================================================================
# Heuristic candidates for the player's NFL team (not the fantasy franchise)
_NFL_TEAM_COL_CANDIDATES = [
    "nfl team", "nfl", "player team", "pro team", "pro team abbrev",
    "team (nfl)", "team_abbrev", "tm", "nflteam"
]

def _normalize_name_series(s: pd.Series) -> pd.Series:
    cleaned = (
        s.astype(str)
         .str.normalize("NFKD")
         .str.encode("ascii", errors="ignore").str.decode("ascii", errors="ignore")
         .str.replace(r"[^\w\s]", "", regex=True)
         .str.replace(r"\s+", " ", regex=True)
         .str.strip()
         .str.lower()
    )
    cleaned = cleaned.str.replace(r"\b(jr|sr|ii|iii|iv|v)\b", "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    return cleaned

@st.cache_data(show_spinner=False)
def _parse_live_picks(file_bytes: Optional[bytes]) -> pd.DataFrame:
    """Tolerant CSV reader for live picks. Returns DataFrame with Player (+ optional Pos, NFLTeam, team_manager)."""
    if not file_bytes:
        return pd.DataFrame()

    # Sniff delimiter; BOM-safe
    buf = io.BytesIO(file_bytes)
    sample = buf.read(8192).decode("utf-8", errors="ignore")
    try:
        dialect = csv.Sniffer().sniff(sample)
        delim = dialect.delimiter
    except Exception:
        delim = ","
    buf.seek(0)
    try:
        df = pd.read_csv(buf, engine="python", sep=delim, encoding="utf-8-sig", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df.columns = [str(c).strip() for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}

    # Player
    name_col = next((lower_map[k] for k in ["player", "player name", "name", "athlete", "full name", "playername"] if k in lower_map), None)
    if name_col is None:
        name_col = df.columns[0]  # fallback

    out = df.copy()
    out["Player"] = df[name_col].astype(str).str.strip()

    # Pos (optional but improves matching)
    pos_col = next((lower_map[k] for k in ["pos", "position"] if k in lower_map), None)
    if pos_col:
        out["Pos"] = _canon_pos_series_ui(df[pos_col])

    # NFL team (optional; DO NOT confuse with franchise)
    nfl_col = None
    for k in _NFL_TEAM_COL_CANDIDATES:
        if k in lower_map:
            nfl_col = lower_map[k]
            break
    if nfl_col:
        out["NFLTeam"] = df[nfl_col].astype(str).str.upper().str.strip()

    # team_manager column (required by design)
    tm_col = lower_map.get("team_manager")  # exact name per requirement (case-insensitive)
    if tm_col:
        out["team_manager"] = df[tm_col].astype(str).str.strip()
    else:
        # keep a placeholder so downstream code can branch gracefully
        out["team_manager"] = pd.NA

    # Drop blanks
    out = out[out["Player"].astype(str).str.strip().ne("")]
    out = out.reset_index(drop=True)
    return out

def _pids_from_live(live_df: pd.DataFrame, model_df: pd.DataFrame) -> Set[str]:
    """
    Map live picks to model PIDs using (name+pos) and (name+NFL team) when available.
    Name-only fallback is used **only when the name is unique** in the model to avoid over-matching.
    """
    if live_df is None or live_df.empty or model_df is None or model_df.empty:
        return set()

    live = live_df.copy()
    mdl = model_df.copy()

    live["__name_key"] = _normalize_name_series(live["Player"])
    mdl["__name_key"]  = _normalize_name_series(mdl["Player"])

    taken: Set[str] = set()

    # 1) Name+Pos
    if "Pos" in live.columns:
        tmp = mdl.merge(live[["__name_key", "Pos"]].drop_duplicates(), on=["__name_key", "Pos"], how="inner")
        taken.update(tmp["PID"].unique().tolist())

    # 2) Name+NFLTeam (model uses 'Team')
    if "NFLTeam" in live.columns:
        tmp = mdl.merge(
            live[["__name_key", "NFLTeam"]].drop_duplicates(),
            left_on=["__name_key", "Team"], right_on=["__name_key", "NFLTeam"], how="inner"
        )
        taken.update(tmp["PID"].unique().tolist())

    # 3) Name-only fallback **only for unique names in the model**
    name_counts = mdl.groupby("__name_key")["PID"].nunique()
    unique_names = set(name_counts[name_counts == 1].index)
    unmatched_live_names = set(live["__name_key"].unique()) - set(mdl[mdl["PID"].isin(taken)]["__name_key"].unique())
    safe_names = list(unmatched_live_names & unique_names)
    if safe_names:
        tmp = mdl[mdl["__name_key"].isin(safe_names)]
        taken.update(tmp["PID"].unique().tolist())

    return taken

def _pids_for_my_team(live_df: pd.DataFrame, model_df: pd.DataFrame, my_team_value: Optional[str]) -> Set[str]:
    """Restrict live picks to rows where team_manager == my_team_value, then map to PIDs."""
    if not my_team_value or live_df is None or live_df.empty or "team_manager" not in live_df.columns:
        return set()
    mask = live_df["team_manager"].astype(str).str.strip().str.casefold() == str(my_team_value).strip().casefold()
    sub = live_df.loc[mask]
    return _pids_from_live(sub, model_df)

def _infer_round_from_live(n_picks_made: int, teams: int, slot: int, rounds: int) -> int:
    """
    Given number of picks recorded, infer the round in which your NEXT pick occurs.
    We find the smallest r with pick_number(r, teams, slot) >= (n_picks_made + 1).
    """
    next_overall = int(n_picks_made) + 1
    for r in range(1, int(rounds) + 1):
        if pick_number(r, int(teams), int(slot)) >= next_overall:
            return r
    return int(rounds)

# =============================================================================
# Sidebar — League, Advanced, Data files, Live CSV (team_manager)
# =============================================================================
with st.sidebar:
    st.subheader("League")
    teams = st.number_input("Teams", 4, 20, 14, 1)
    slot = st.number_input("Your pick # (1..Teams)", 1, int(teams), 3, 1)
    rounds = st.number_input("Rounds", 8, 25, 16, 1)

    st.caption("Starters (affects replacement baselines)")
    c1, c2, c3 = st.columns(3)
    with c1:
        s_qb = st.number_input("QB", 0, 3, 1, 1)
        s_k = st.number_input("K", 0, 2, 1, 1)
    with c2:
        s_rb = st.number_input("RB", 0, 5, 2, 1)
        s_dst = st.number_input("DST", 0, 2, 1, 1)
    with c3:
        s_wr = st.number_input("WR", 0, 5, 2, 1)
        s_te = st.number_input("TE", 0, 3, 1, 1)
    s_fx = st.number_input("FLEX (RB/WR/TE)", 0, 3, 1, 1)

    with st.expander("Advanced (availability & risk)"):
        use_adp_sigma = st.checkbox("Use σ from ADP if present", True)
        use_linear_sigma = st.checkbox("If no σ, use σ = a + b·ADP (else constant)", True)
        sigma_const = st.number_input("σ (constant)", 1.0, 50.0, 12.0, 0.5)
        sigma_a = st.number_input("σ = a + b·ADP (a)", 0.0, 20.0, 3.0, 0.5)
        sigma_b = st.number_input("σ = a + b·ADP (b)", 0.0, 1.0, 0.04, 0.01)
        sigma_min = st.number_input("σ min", 1.0, 50.0, 6.0, 0.5)
        sigma_max = st.number_input("σ max", 5.0, 80.0, 26.0, 0.5)
        st.session_state["risk_lambda"] = st.number_input("Risk penalty λ (season SD)", 0.00, 1.00, 0.15, 0.01)

    # -------------------------------------------------------------------------
    # Projections/ADP from ./data
    # -------------------------------------------------------------------------
    st.divider()
    st.subheader("Data source (projections & ADP)")
    data_dir = _resolve_data_dir()
    st.caption(f"Using data folder:\n\n`{data_dir}`")
    files = _discover_files(data_dir)

    def _status_line(label: str, p: Optional[Path]):
        tick = "✅" if p else "❌"
        st.write(f"{tick} **{label}** — {p.name if p else 'not found'}")

    _status_line("FLX projections (RB/WR/TE) — required", files.get("flx"))
    _status_line("QB projections (optional)", files.get("qb"))
    _status_line("Kicker projections (optional)", files.get("k"))
    _status_line("DST projections (optional)", files.get("dst"))
    _status_line("Overall ADP (optional)", files.get("adp"))

    if st.button("🔄 Refresh projections/ADP", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    with st.expander("Manual override (optional)"):
        qb_up  = st.file_uploader("QB projections (CSV)", type="csv", key="upQB")
        flx_up = st.file_uploader("FLX projections (RB/WR/TE) — required if not in folder", type="csv", key="upFLX")
        k_up   = st.file_uploader("Kicker projections (CSV)", type="csv", key="upK")
        dst_up = st.file_uploader("DST projections (CSV)", type="csv", key="upDST")
        adp_up = st.file_uploader("Overall ADP (CSV)", type="csv", key="upADP")

    # -------------------------------------------------------------------------
    # Live picks from ./data/liveData (file + required team_manager selector)
    # -------------------------------------------------------------------------
    st.divider()
    st.subheader("Live draft picks")
    live_dir = _resolve_live_dir()

    live_files = sorted(list(live_dir.glob("*.csv")), key=lambda p: p.stat().st_mtime, reverse=True)
    preferred = _find_by_exact(live_dir, "espn_draft_picks.csv")
    live_options = [p.name for p in live_files]
    default_idx = 0
    if preferred:
        try:
            default_idx = live_options.index(preferred.name)
        except ValueError:
            pass

    selected_live_name = st.selectbox("Choose live CSV", options=live_options or ["(none found)"], index=default_idx if live_options else 0)
    live_csv = (live_dir / selected_live_name) if live_options else None

    if live_csv and live_csv.exists():
        ts = datetime.fromtimestamp(live_csv.stat().st_mtime)
        st.write(f"✅ Using: **{live_csv.name}**  \n_Updated: {ts.strftime('%Y-%m-%d %H:%M:%S')}_")
    else:
        st.write("❌ No live picks CSV found in `data/liveData/` (will still run).")

    if st.button("🔄 Refresh live picks", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# =============================================================================
# Engine config objects
# =============================================================================
league = LeagueConfig(
    teams=int(teams), slot=int(slot), rounds=int(rounds),
    starters={"QB": int(s_qb), "RB": int(s_rb), "WR": int(s_wr), "TE": int(s_te),
              "FLEX": int(s_fx), "K": int(s_k), "DST": int(s_dst)},
)
var = VarianceModel(
    use_adp_sigma=use_adp_sigma, use_linear_sigma=use_linear_sigma,
    sigma_const=sigma_const, sigma_a=sigma_a, sigma_b=sigma_b,
    sigma_min=sigma_min, sigma_max=sigma_max,
)
inj = InjuryModel()
risk_by_pos = {"QB": 0.45, "RB": 0.55, "WR": 0.50, "TE": 0.50, "K": 0.40, "DST": 0.30}
cv_by_pos   = {"QB": 0.35, "RB": 0.55, "WR": 0.60, "TE": 0.60, "K": 0.30, "DST": 0.25}

# =============================================================================
# Cached wrappers (UI-only)
# =============================================================================
@st.cache_data(show_spinner=False)
def _load_proj_cached(qb_b: Optional[bytes], flx_b: Optional[bytes],
                      k_b: Optional[bytes], dst_b: Optional[bytes]) -> pd.DataFrame:
    return load_fp_uploads(qb_b, flx_b, k_b, dst_b)

@st.cache_data(show_spinner=False)
def _build_model_cached(proj: pd.DataFrame, league_: LeagueConfig, var_: VarianceModel, inj_: InjuryModel,
                        risk_map: dict, cv_map: dict, adp_: Optional[pd.DataFrame]) -> pd.DataFrame:
    return build_model_df(proj, league_, var_, inj_, risk_map, cv_map, adp_df=adp_)

# =============================================================================
# Load projections/ADP (auto + manual override)
# =============================================================================
auto_qb_b, auto_flx_b, auto_k_b, auto_dst_b, auto_adp_b = _auto_read_bytes(files)
qb_b  = (qb_up.getvalue()  if 'qb_up'  in globals() and qb_up  else auto_qb_b)
flx_b = (flx_up.getvalue() if 'flx_up' in globals() and flx_up else auto_flx_b)
k_b   = (k_up.getvalue()   if 'k_up'   in globals() and k_up   else auto_k_b)
dst_b = (dst_up.getvalue() if 'dst_up' in globals() and dst_up else auto_dst_b)
adp_b = (adp_up.getvalue() if 'adp_up' in globals() and adp_up else auto_adp_b)

proj_df = _load_proj_cached(qb_b, flx_b, k_b, dst_b)

# Robust ADP parse; ensure ["Player","Pos","ADP"] exist (NaN ok)
adp_df = _robust_parse_adp(adp_b)
adp_for_engine = adp_df if adp_df is not None else None

if proj_df is None or proj_df.empty:
    st.info(
        "📁 **No projections loaded.**\n\n"
        "- Ensure **data/** has at least the FLX projections CSV.\n"
        "- Optional: QB, K, DST projections and an Overall ADP CSV.\n"
        "- Or use **Sidebar → Manual override** to upload files directly."
    )
    st.stop()

# Build model DataFrame (engine logic; PID = Player|Team|Pos)
model_df = _build_model_cached(proj_df, league, var, inj, risk_by_pos, cv_by_pos, adp_for_engine)
if model_df is None or model_df.empty:
    st.error("Model data failed to build. Check your files in the data folder or your uploads.")
    st.stop()

# =============================================================================
# Load live picks; compute availability and my picks (team_manager)
# =============================================================================
live_bytes = _read_bytes(live_csv) if live_csv and live_csv.exists() else None
live_df = _parse_live_picks(live_bytes)

# team_manager selection (required path)
with st.sidebar:
    st.subheader("My team (from team_manager)")
    if not live_df.empty and "team_manager" in live_df.columns and live_df["team_manager"].notna().any():
        mgr_values = (
            live_df["team_manager"].astype(str).str.strip()
            .replace("", np.nan).dropna().drop_duplicates().sort_values().tolist()
        )
        # default to previous selection if still present; else first
        default_val = st.session_state.get("my_team_val")
        if default_val not in mgr_values and mgr_values:
            default_val = mgr_values[0]
        if mgr_values:
            my_team_val = st.selectbox("Select your **team_manager**", options=mgr_values,
                                       index=mgr_values.index(default_val) if default_val in mgr_values else 0)
            st.session_state["my_team_val"] = my_team_val
        else:
            st.info("`team_manager` column has no values.")
            my_team_val = None
    elif not live_df.empty:
        st.error("Live CSV loaded, but **team_manager** column was not found. Please include this column.")
        my_team_val = None
    else:
        st.caption("No live CSV loaded yet.")
        my_team_val = None

# Unavailable = all live picks; My picks = those for your team_manager
live_taken_pids: Set[str] = _pids_from_live(live_df, model_df)
my_pids: Set[str] = _pids_for_my_team(live_df, model_df, my_team_val)

# Infer current round from how many picks are logged
n_picks_made = int(live_df.shape[0]) if not live_df.empty else 0
round_now = _infer_round_from_live(n_picks_made, league.teams, league.slot, league.rounds)
pick_now = pick_number(round_now, league.teams, league.slot)
next_round = min(league.rounds, round_now + 1)
pick_next = pick_number(next_round, league.teams, league.slot)

# =============================================================================
# Header — context
# =============================================================================
c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
with c1: st.metric("Picks logged", n_picks_made)
with c2: st.metric("Current round", round_now)
with c3: st.metric("Your slot", league.slot)
with c4: st.markdown(f"**Your next pick:** Round {round_now}, Overall #{pick_now}  \n**Next after that:** Round {next_round}, Overall #{pick_next}")

# Show a glance of the last few live picks (optional)
if not live_df.empty:
    st.caption("Recent live picks (from file)")
    show_cols = [c for c in ["Player", "Pos", "NFLTeam", "team_manager"] if c in live_df.columns]
    st.dataframe(live_df.tail(8)[show_cols] if show_cols else live_df.tail(8), hide_index=True, use_container_width=True)

# =============================================================================
# Targets — compute & render (availability from live CSV)
# =============================================================================
topN = int(st.slider("How many recommendations to show", 5, 30, 10, 1))

unavailable: Set[str] = set(live_taken_pids)
targets = compute_targets(
    model_df, round_now, league, var, top_n=topN,
    unavailable_pids=unavailable,
    risk_lambda=float(st.session_state.get("risk_lambda", 0.15)),
    my_pids=set(my_pids),  # NEW: roster-aware ΔEV
)

st.markdown("### Recommended now")
total_avail = int((~model_df["PID"].isin(unavailable)).sum())
st.caption(f"Showing **{len(targets)}** of **{total_avail}** available players.")

for _, row in targets.iterrows():
    p = row["Player"]; pos = row["Pos"]; team = row.get("Team", "")  # model team (NFL)
    adp_val = row.get("ADP", np.nan)
    pav = float(row["PAvailNext"]); val = float(row["ValueNow"])
    ra = float(row["RiskAdj"]); pg = float(row["PerGame"])

    parts = [f"**{p}** — {pos} {team}"]
    if pd.notna(adp_val):
        parts.append(f"ADP {float(adp_val):.1f}")
    parts.extend([f"P@Next {pav:.2f}", f"ΔPtsNow(season) {val:.1f}", f"RiskScore {ra:.1f}", f"Pts/G {pg:.1f}"])
    st.write(" | ".join(parts))

# =============================================================================
# My team’s picks (derived from live CSV via team_manager)
# =============================================================================
st.divider()
st.markdown("#### My picks (from live CSV)")
if my_pids:
    mine_df = model_df[model_df["PID"].isin(my_pids)][["Player","Pos","Team","PerGame","PreWeeklyEV_base"]]
    mine_df = mine_df.sort_values(["Pos","PreWeeklyEV_base"], ascending=[True, False])
    st.dataframe(
        mine_df.rename(columns={"PreWeeklyEV_base":"ΔPtsNow(season)"}),
        hide_index=True, use_container_width=True
    )
else:
    st.caption("No picks found yet for your selected **team_manager** (or none selected).")

# =============================================================================
# Diagnostics (optional)
# =============================================================================
with st.expander("Diagnostics"):
    # Available counts by position
    avail_mask = ~model_df["PID"].isin(unavailable)
    pos_counts = model_df.loc[avail_mask, "Pos"].value_counts().sort_index()
    adp_coverage = (pd.notna(model_df.get("ADP", np.nan)) & avail_mask).sum()
    st.write("**Available by position:**")
    st.write(pos_counts.to_frame("Count"))
    st.write(f"**ADP present for** {adp_coverage} of {int(avail_mask.sum())} available rows.")
    if adp_df is not None:
        st.caption(f"ADP rows loaded: {len(adp_df)} (non‑null ADP: {int(pd.notna(adp_df['ADP']).sum())})")

with st.expander("What do these mean?", expanded=False):
    st.markdown("""
- **ADP** — Average Draft Position from your ADP CSV (or inferred from projections order).
- **P@Next** — Probability the player is *still available at your next pick* (snake‑aware, adjusted for runs).
- **ΔPtsNow (season)** — **Roster-aware** season points **above replacement** you add to your lineup if you draft the player now.
- **RiskScore** — Ranking score = ΔPtsNow − λ·SeasonSD + 0.5·Regret.
- **Pts/G** — Projected points per game.
""")
