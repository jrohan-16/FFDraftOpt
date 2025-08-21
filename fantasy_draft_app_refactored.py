# -*- coding: utf-8 -*-
r"""
Fantasy Draft Console — Streamlit UI (auto data folder; tolerant CSV loading)

- Automatically loads CSVs from a ./data directory next to this script.
- Manual upload remains as an optional fallback (and overrides auto if used).
- NEW: "Best-effort" CSV loader for ADP to tolerate messy lines (extra commas).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, List

import io
import numpy as np
import pandas as pd
import streamlit as st

from fantasy_draft_engine import (
    LeagueConfig, VarianceModel, InjuryModel,
    load_fp_uploads, load_adp_upload,
    build_model_df, compute_targets, pick_number
)

st.set_page_config(page_title="Fantasy Draft Console (Auto Data Folder)", layout="centered")
st.title("Draft Console (Auto Data Folder)")

# ----------------------------
# Path helpers
# ----------------------------
def _app_root() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()

def _resolve_data_dir() -> Path:
    candidates = [
        _app_root() / "data",
        Path.cwd() / "data",
    ]
    for d in candidates:
        if d.exists() and d.is_dir():
            return d
    d = candidates[0]
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
    must_not = must_not or []
    hits: List[Path] = []
    for p in data_dir.glob("*.csv"):
        name = p.name.lower()
        if all(k.lower() in name for k in must_have) and all(n.lower() not in name for n in must_not):
            hits.append(p)
    return _pick_latest(hits)

def _discover_files(data_dir: Path) -> Dict[str, Optional[Path]]:
    exact = {
        "qb":  "FantasyPros_Fantasy_Football_Projections_QB.csv",
        "flx": "FantasyPros_Fantasy_Football_Projections_FLX.csv",
        "k":   "FantasyPros_Fantasy_Football_Projections_K.csv",
        "dst": "FantasyPros_Fantasy_Football_Projections_DST.csv",
        "adp": "FantasyPros_2025_Overall_ADP_Rankings.csv",
    }
    out: Dict[str, Optional[Path]] = {k: _find_by_exact(data_dir, v) for k, v in exact.items()}
    if out["flx"] is None:
        out["flx"] = (_find_by_keywords(data_dir, ["flx"]) or
                      _find_by_keywords(data_dir, ["rb", "wr", "te"]))
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
        out["adp"] = (_find_by_keywords(data_dir, ["2025", "adp"]) or
                      _find_by_keywords(data_dir, ["overall", "adp"]) or
                      _find_by_keywords(data_dir, ["adp"]))
    return out

def _read_bytes(p: Optional[Path]) -> Optional[bytes]:
    if p is None:
        return None
    try:
        return p.read_bytes()
    except Exception:
        return None

def _auto_read_bytes(files: Dict[str, Optional[Path]]) -> Tuple[Optional[bytes], Optional[bytes], Optional[bytes], Optional[bytes], Optional[bytes]]:
    qb_b = _read_bytes(files.get("qb"))
    flx_b = _read_bytes(files.get("flx"))
    k_b = _read_bytes(files.get("k"))
    dst_b = _read_bytes(files.get("dst"))
    adp_b = _read_bytes(files.get("adp"))
    return qb_b, flx_b, k_b, dst_b, adp_b

# ----------------------------
# Sidebar config
# ----------------------------
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

    st.divider()
    st.subheader("Data source")
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

    if st.button("🔄 Refresh data from folder", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    with st.expander("Manual override (optional)"):
        qb_up = st.file_uploader("QB projections (CSV)", type="csv", key="upQB")
        flx_up = st.file_uploader("FLX projections (RB/WR/TE) — required if not in folder", type="csv", key="upFLX")
        k_up = st.file_uploader("Kicker projections (CSV)", type="csv", key="upK")
        dst_up = st.file_uploader("DST projections (CSV)", type="csv", key="upDST")
        adp_up = st.file_uploader("Overall ADP (CSV)", type="csv", key="upADP")

# ----------------------------
# Engine config objects
# ----------------------------
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

# ----------------------------
# Cached data loaders
# ----------------------------
@st.cache_data(show_spinner=False)
def _load_proj_cached(qb_b: Optional[bytes], flx_b: Optional[bytes],
                      k_b: Optional[bytes], dst_b: Optional[bytes]) -> pd.DataFrame:
    return load_fp_uploads(qb_b, flx_b, k_b, dst_b)

def _robust_parse_adp_bytes(adp_bytes: bytes) -> Optional[pd.DataFrame]:
    """
    Try multiple tolerant parsing strategies to salvage an ADP CSV that has
    stray commas or inconsistent column counts. Returns a minimal ADP DataFrame
    with columns ['Player','Pos','ADP'] when possible; otherwise None.
    """
    if not adp_bytes:
        return None
    # Attempt 1: python engine, keep bad lines but warn
    try:
        df = pd.read_csv(io.BytesIO(adp_bytes), engine="python",
                         sep=",", quotechar='"', escapechar="\\",
                         on_bad_lines="warn")
    except Exception:
        # Attempt 2: sniff delimiter
        b = io.BytesIO(adp_bytes)
        sample = b.read(8192).decode("utf-8", errors="ignore")
        import csv as _csv
        try:
            dialect = _csv.Sniffer().sniff(sample)
            delim = dialect.delimiter
        except Exception:
            delim = ","
        b.seek(0)
        try:
            df = pd.read_csv(b, engine="python", sep=delim, on_bad_lines="skip")
        except Exception:
            return None

    # Normalize & reduce to the needed columns
    df.columns = [str(c).strip() for c in df.columns]
    name_col = next((c for c in df.columns if c.lower() in ("player","player name","name","playername")), None)
    pos_col  = next((c for c in df.columns if c.lower() in ("pos","position")), None)
    adp_col  = next((c for c in df.columns if c.lower() in ("adp","overall adp","overall_adp","avg. draft position","average draft position","avg draft position")), None)

    if name_col is None:
        # Assume first column is the name if not explicitly labeled
        name_col = df.columns[0]

    out = pd.DataFrame({
        "Player": df[name_col].astype(str).str.strip(),
        "Pos": df[pos_col] if pos_col in df.columns else pd.NA,
        "ADP": pd.to_numeric(df[adp_col], errors="coerce") if adp_col in df.columns else pd.NA
    })
    out = out[~out["Player"].isna() & out["Player"].astype(str).str.len().astype(int).gt(0)]
    if out.empty:
        return None
    return out

@st.cache_data(show_spinner=False)
def _load_adp_best_effort(adp_b: Optional[bytes]) -> Tuple[Optional[pd.DataFrame], bool]:
    if not adp_b:
        return None, False
    try:
        return load_adp_upload(adp_b), False
    except Exception:
        return _robust_parse_adp_bytes(adp_b), True

# ----------------------------
# Load data (auto + manual override)
# ----------------------------
auto_qb_b, auto_flx_b, auto_k_b, auto_dst_b, auto_adp_b = _auto_read_bytes(files)
qb_b  = (qb_up.getvalue()  if qb_up  else auto_qb_b)
flx_b = (flx_up.getvalue() if flx_up else auto_flx_b)
k_b   = (k_up.getvalue()   if k_up   else auto_k_b)
dst_b = (dst_up.getvalue() if dst_up else auto_dst_b)
adp_b = (adp_up.getvalue() if adp_up else auto_adp_b)

proj_df = _load_proj_cached(qb_b, flx_b, k_b, dst_b)
adp_df, adp_fallback_used  = _load_adp_best_effort(adp_b)

if proj_df is None or proj_df.empty:
    st.info(
        "📁 **No projections loaded.**\n\n"
        "- Make sure the folder **`data/`** next to this app contains at least the "
        "**FLX projections** file (RB/WR/TE). Common filename:\n"
        "  `FantasyPros_Fantasy_Football_Projections_FLX.csv`\n"
        "- Optional files: QB, K, DST projections and an Overall ADP CSV.\n"
        "- Or open **Sidebar → Manual override** to upload files directly."
    )
    st.stop()

if adp_b and adp_df is None:
    st.warning("ADP file could not be parsed; proceeding **without ADP**. Availability & sigma-from-ADP features will be less accurate.")
elif adp_b and adp_fallback_used:
    st.info("ADP file had format issues; loaded with a tolerant parser. Some bad rows may have been skipped.")

model_df = build_model_df(proj_df, league, var, inj, risk_by_pos, cv_by_pos, adp_df=adp_df)
if model_df is None or model_df.empty:
    st.error("Model data failed to build. Check your files in the data folder or your uploads.")
    st.stop()

# ----------------------------
# Session state
# ----------------------------
if "taken" not in st.session_state: st.session_state["taken"] = set()
if "mine"  not in st.session_state: st.session_state["mine"]  = set()
if "history" not in st.session_state: st.session_state["history"] = []
if "current_round" not in st.session_state: st.session_state["current_round"] = 1

# ----------------------------
# Header
# ----------------------------
round_now = int(st.session_state["current_round"])
pick_now = pick_number(round_now, league.teams, league.slot)
next_round = min(league.rounds, round_now + 1)
pick_next = pick_number(next_round, league.teams, league.slot)

c1, c2, c3 = st.columns([1, 1, 2])
with c1: st.metric("Current round", round_now)
with c2: st.metric("Your pick #", pick_now)
with c3: st.markdown(f"**Next pick:** Round {next_round}, Overall #{pick_next}")

# ----------------------------
# Targets
# ----------------------------
topN = int(st.slider("How many recommendations to show", 5, 30, 10, 1))
unavailable = st.session_state["taken"].union(st.session_state["mine"])
targets = compute_targets(
    model_df, round_now, league, var, top_n=topN,
    unavailable_pids=unavailable, risk_lambda=float(st.session_state.get("risk_lambda", 0.15))
)

st.markdown("### Recommended now")
total_avail = int((~model_df["PID"].isin(unavailable)).sum())
st.caption(f"Showing **{len(targets)}** of **{total_avail}** available players.")

for _, row in targets.iterrows():
    p = row["Player"]; pos = row["Pos"]; team = row.get("Team",""); pid = row["PID"]
    adp = float(row.get("ADP", float('nan'))) if pd.notna(row.get("ADP", pd.NA)) else float('nan')
    pav = float(row["PAvailNext"]); val = float(row["ValueNow"])
    ra = float(row["RiskAdj"]); pg = float(row["PerGame"])

    b1, b2, b3, b4 = st.columns([5, 1.5, 1.5, 1.2])
    with b1:
        parts = [f"**{p}** — {pos} {team}"]
        if pd.notna(adp):
            parts.append(f"ADP {adp:.1f}")
        parts.append(f"P@Next {pav:.2f}")
        parts.append(f"ΔPtsNow(season) {val:.1f}")
        parts.append(f"RiskScore {ra:.1f}")
        parts.append(f"Pts/G {pg:.1f}")
        st.write(" | ".join(parts))
    with b2:
        if st.button("Draft", key=f"mine_btn_{pid}"):
            st.session_state["mine"].add(pid)
            st.session_state["taken"].add(pid)
            st.session_state["history"].append(("draft", pid))
            st.session_state["current_round"] = min(league.rounds, round_now + 1)
            st.rerun()
    with b3:
        if st.button("Mark taken", key=f"taken_btn_{pid}"):
            st.session_state["taken"].add(pid)
            st.session_state["history"].append(("taken", pid))
            st.rerun()
    with b4:
        if st.button("Hide", key=f"hide_btn_{pid}"):
            st.session_state["taken"].add(pid)
            st.session_state["history"].append(("hide", pid))
            st.rerun()

# ----------------------------
# Roster & controls
# ----------------------------
st.divider()
st.markdown("#### My picks this draft")
mine_df = model_df[model_df["PID"].isin(st.session_state["mine"])][["Player","Pos","Team","PerGame","PreWeeklyEV_base"]]
if mine_df.empty:
    st.caption("No picks yet.")
else:
    mine_df = mine_df.sort_values(["Pos","PreWeeklyEV_base"], ascending=[True, False])
    st.dataframe(
        mine_df.rename(columns={"PreWeeklyEV_base":"ΔPtsNow(season)"}),
        hide_index=True, use_container_width=True
    )

cA, cB = st.columns([1, 1])
with cA:
    if st.button("↩︎ Undo last action", type="secondary"):
        if st.session_state["history"]:
            op, pid = st.session_state["history"].pop()
            if op == "draft":
                st.session_state["mine"].discard(pid)
                st.session_state["taken"].discard(pid)
                st.session_state["current_round"] = max(1, round_now - 1)
            elif op in ("taken", "hide"):
                st.session_state["taken"].discard(pid)
            st.rerun()
with cB:
    if st.button("🧹 Reset board / round", type="secondary"):
        st.session_state["taken"].clear()
        st.session_state["mine"].clear()
        st.session_state["history"].clear()
        st.session_state["current_round"] = 1
        st.rerun()

with st.expander("What do these mean?", expanded=False):
    st.markdown("""
- **ADP** — Average Draft Position from your ADP CSV (or inferred from projections order).
- **P@Next** — Probability the player is *still available at your next pick* (snake-aware).
- **ΔPtsNow (season)** — Expected **season** points **above replacement** if you draft the player now.
- **RiskScore** — Ranking score = ΔPtsNow − λ·SeasonSD + 0.5·Regret.
- **Pts/G** — Projected points per game.
""")
