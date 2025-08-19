# -*- coding: utf-8 -*-
"""
Fantasy Draft Console — Upload-Only Minimal UI (v3.5)

Key fixes in this version
-------------------------
• **Upload-only** (no defaults). You must upload projections & ADP.
• **Fix crash**: replaced `.clip(lower=1.0)` on numpy arrays (now uses `np.maximum(..., 1.0)`).
• **Shows fallers** above your pick to catch slips.
• **Board progresses** on Draft/Mark taken/Hide (state + rerun).
• **Exact slider count** for recommendations.
• **Snake‑aware P@Next** + FLEX‑aware replacement + risk-adjusted score with one-pick lookahead.

Run:
    streamlit run fantasy_draft_app.py
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# =============================================================================
# Utility math
# =============================================================================

def normal_cdf(z: np.ndarray | float) -> np.ndarray | float:
    z = np.asarray(z, dtype=float)
    t = 1.0 / (1.0 + 0.5 * np.abs(z))
    tau = t * np.exp(
        -z*z
        - 1.26551223
        + 1.00002368 * t
        + 0.37409196 * t**2
        + 0.09678418 * t**3
        - 0.18628806 * t**4
        + 0.27886807 * t**5
        - 1.13520398 * t**6
        + 1.48851587 * t**7
        - 0.82215223 * t**8
        + 0.17087277 * t**9
    )
    erf = 1.0 - tau
    erf = np.where(z >= 0, erf, -erf)
    return 0.5 * (1.0 + erf)

# =============================================================================
# Config
# =============================================================================

@dataclass
class LeagueConfig:
    teams: int = 14
    slot: int = 3
    rounds: int = 16
    starters: Dict[str, int] | None = None
    flex_shares: Dict[str, float] | None = None
    use_predraft_startshare: bool = True
    stream_boost_k: float = 0.0
    stream_boost_dst: float = 0.0
    def __post_init__(self):
        if self.starters is None:
            self.starters = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "DST": 1}
        if self.flex_shares is None:
            self.flex_shares = {"RB": 0.45, "WR": 0.50, "TE": 0.05}

@dataclass
class VarianceModel:
    use_adp_sigma: bool = True
    use_linear_sigma: bool = True
    sigma_const: float = 12.0
    sigma_a: float = 3.0
    sigma_b: float = 0.04
    sigma_min: float = 6.0
    sigma_max: float = 26.0

@dataclass
class InjuryModel:
    risk_alpha: float = 1.0
    miss_at_risk05: Dict[str, float] | None = None
    episode_len: Dict[str, float] | None = None
    def __post_init__(self):
        if self.miss_at_risk05 is None:
            self.miss_at_risk05 = {"QB": 1.0, "RB": 2.5, "WR": 2.0, "TE": 2.0, "K": 0.8, "DST": 0.2}
        if self.episode_len is None:
            self.episode_len = {"QB": 1.5, "RB": 2.5, "WR": 2.0, "TE": 2.0, "K": 1.0, "DST": 0.5}

DEFAULT_START_SHARE = {"QB": 0.95, "RB": 0.65, "WR": 0.60, "TE": 0.50, "K": 0.95, "DST": 0.95}
DEFAULT_CV = {"QB": 0.35, "RB": 0.55, "WR": 0.60, "TE": 0.60, "K": 0.30, "DST": 0.25}

# =============================================================================
# CSV ingestion (upload-only)
# =============================================================================

def cols_lower(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def first_present(cols: List[str], options: List[str]) -> Optional[str]:
    for o in options:
        if o in cols:
            return o
    lower = {c.lower(): c for c in cols}
    for o in options:
        if o.lower() in lower:
            return lower[o.lower()]
    return None

def find_team_col(df: pd.DataFrame) -> Optional[str]:
    return first_present(df.columns, ["team", "tm", "nfl team"])

def find_bye_col(df: pd.DataFrame) -> Optional[str]:
    return first_present(df.columns, ["bye", "bye week", "bye week number", "bye_week"])

def find_fpts_col(df: pd.DataFrame) -> Optional[str]:
    return first_present(df.columns, ["fpts", "fp", "points", "proj", "projected pts", "fantasy pts", "fpts (proj)"])

@st.cache_data(show_spinner=False)
def _load_fp_from_bytes(content: bytes, pos_hint: Optional[str]) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(content))
    df = cols_lower(df)

    player_col = first_present(df.columns, ["player", "name", "player name"]) or list(df.columns)[0]
    team_col = find_team_col(df) or ""
    bye_col = find_bye_col(df)

    fpts_col = find_fpts_col(df)
    if fpts_col is None:
        df["__projpts__"] = 0.0
    else:
        df["__projpts__"] = pd.to_numeric(df[fpts_col], errors="coerce").fillna(0.0)

    if pos_hint is None:
        pos_col = first_present(df.columns, ["pos", "position"])
        if pos_col is None:
            raise ValueError("Could not infer 'Pos' from FLX upload. Please include a 'Pos' column.")
        pos_series = df[pos_col].astype(str).str.upper().str.replace(" ", "", regex=False).replace({"DEF": "DST"})
    else:
        pos_series = pos_hint

    out = pd.DataFrame({
        "Player": df[player_col].astype(str).str.strip(),
        "Pos": pos_series if isinstance(pos_series, str) else pos_series.astype(str),
        "Team": df[team_col].astype(str).str.upper().str.strip() if team_col in df.columns else "",
        "Bye": df[bye_col] if (bye_col is not None and bye_col in df.columns) else np.nan,
        "ProjPts": df["__projpts__"].astype(float),
    })

    # Carry ADP/ECR if present
    for adp_col in ["adp", "avg", "ecr"]:
        if adp_col in df.columns:
            out[adp_col.upper()] = pd.to_numeric(df[adp_col], errors="coerce")

    return out

@st.cache_data(show_spinner=False)
def load_fp_uploads(qb_file, flx_file, k_file, dst_file) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    if qb_file is not None:
        frames.append(_load_fp_from_bytes(qb_file.getvalue(), pos_hint="QB"))
    if flx_file is not None:
        frames.append(_load_fp_from_bytes(flx_file.getvalue(), pos_hint=None))
    if k_file is not None:
        frames.append(_load_fp_from_bytes(k_file.getvalue(), pos_hint="K"))
    if dst_file is not None:
        frames.append(_load_fp_from_bytes(dst_file.getvalue(), pos_hint="DST"))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

@st.cache_data(show_spinner=False)
def load_adp_upload(adp_file) -> pd.DataFrame:
    if adp_file is None:
        return pd.DataFrame()
    df = pd.read_csv(adp_file)
    d = cols_lower(df)

    player_col = first_present(d.columns, ["player", "name"]) or list(d.columns)[0]
    pos_col = first_present(d.columns, ["pos", "position"])
    adp_col = first_present(d.columns, ["adp", "avg", "average", "overall"])
    stdev_col = first_present(d.columns, ["stdev", "std dev", "stddev", "std", "sd"])
    best_col = first_present(d.columns, ["best", "best pick", "min", "min pick"])
    worst_col = first_present(d.columns, ["worst", "worst pick", "max", "max pick"])

    out = pd.DataFrame({"Player": d[player_col].astype(str).str.strip()})
    if pos_col is not None:
        out["Pos"] = d[pos_col].astype(str).str.upper().str.replace(" ", "", regex=False).replace({"DEF": "DST"})
    out["ADP"] = pd.to_numeric(d[adp_col], errors="coerce")

    sig = None
    if stdev_col is not None and stdev_col in d.columns:
        sig = pd.to_numeric(d[stdev_col], errors="coerce")
    elif best_col is not None and worst_col is not None:
        bw = pd.to_numeric(d[worst_col], errors="coerce") - pd.to_numeric(d[best_col], errors="coerce")
        sig = bw / 4.0
    if sig is not None:
        out["SigmaADP"] = sig

    return out

# =============================================================================
# Modeling
# =============================================================================

def markov_proj_g(risk: np.ndarray, pos: np.ndarray, inj: InjuryModel) -> np.ndarray:
    out = np.zeros_like(risk, dtype=float)
    for p in np.unique(pos):
        m = inj.miss_at_risk05.get(p, 2.0)
        L = inj.episode_len.get(p, 2.0)
        h05 = (m / 17.0) / (1.0 - m / 17.0) / L
        mask = pos == p
        rr = np.clip(risk[mask], 0.05, 0.95)
        h = h05 * np.power(rr / 0.5, inj.risk_alpha)
        out[mask] = 17.0 / (1.0 + h * L)
    return out

@st.cache_data(show_spinner=False)
def build_model_df(
    df: pd.DataFrame,
    league: LeagueConfig,
    var: VarianceModel,
    inj: InjuryModel,
    risk_by_pos: Dict[str, float],
    cv_by_pos: Dict[str, float],
    adp_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()

    if "Risk" not in d.columns:
        d["Risk"] = d["Pos"].map(risk_by_pos).fillna(0.5)
    d["Risk"] = pd.to_numeric(d["Risk"], errors="coerce").fillna(0.5).clip(0.05, 0.95)

    # Merge ADP
    if adp_df is not None and not adp_df.empty:
        if "Pos" in adp_df.columns and adp_df["Pos"].notna().any():
            d = d.merge(adp_df[["Player", "Pos", "ADP"] + (["SigmaADP"] if "SigmaADP" in adp_df.columns else [])],
                        on=["Player", "Pos"], how="left", suffixes=("", "_adpfile"))
        else:
            cols = ["Player", "ADP"] + (["SigmaADP"] if "SigmaADP" in adp_df.columns else [])
            d = d.merge(adp_df[cols], on="Player", how="left", suffixes=("", "_adpfile"))
        if "ADP_adpfile" in d.columns:
            d["ADP"] = d["ADP"].combine_first(d["ADP_adpfile"])
            d.drop(columns=["ADP_adpfile"], inplace=True, errors="ignore")
        if "SigmaADP_adpfile" in d.columns and "SigmaADP" not in d.columns:
            d["SigmaADP"] = d["SigmaADP_adpfile"]
            d.drop(columns=["SigmaADP_adpfile"], inplace=True, errors="ignore")

    if "ADP" not in d.columns or d["ADP"].isna().all():
        d = d.sort_values(["ProjPts"], ascending=False).reset_index(drop=True)
        d["ADP"] = (np.arange(1, len(d) + 1)).astype(float)

    # ProjG / PerGame / WeeklySD
    pos_arr = d["Pos"].values.astype(str)
    projg = markov_proj_g(d["Risk"].values.astype(float), pos_arr, inj)
    d["ProjG"] = np.maximum(projg, 1.0)  # FIX: np.maximum instead of .clip(lower=...)
    d["PerGame"] = (d["ProjPts"] / d["ProjG"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    d["WeeklySD"] = d["PerGame"] * d["Pos"].map(cv_by_pos).fillna(0.5)

    # Replacement PPG (FLEX-aware)
    starters = league.starters
    flex_rb = league.teams * starters.get("FLEX", 0) * (league.flex_shares or {}).get("RB", 0.45)
    flex_wr = league.teams * starters.get("FLEX", 0) * (league.flex_shares or {}).get("WR", 0.50)
    flex_te = league.teams * starters.get("FLEX", 0) * (league.flex_shares or {}).get("TE", 0.05)
    league_starters = {
        "QB": league.teams * starters.get("QB", 0),
        "RB": league.teams * starters.get("RB", 0) + flex_rb,
        "WR": league.teams * starters.get("WR", 0) + flex_wr,
        "TE": league.teams * starters.get("TE", 0) + flex_te,
        "K": league.teams * starters.get("K", 0),
        "DST": league.teams * starters.get("DST", 0),
    }

    repl_ppg: Dict[str, float] = {}
    for p in ["QB", "RB", "WR", "TE", "K", "DST"]:
        sub = d[d["Pos"] == p].copy()
        if sub.empty:
            repl_ppg[p] = 0.0
            continue
        sub = sub.sort_values("PerGame", ascending=False)
        brank = int(max(1, min(len(sub), round(league_starters.get(p, 0)))))
        repl_ppg[p] = float(sub.iloc[brank - 1]["PerGame"])
    repl_ppg["K"]  = repl_ppg.get("K", 0.0)  + league.stream_boost_k
    repl_ppg["DST"]= repl_ppg.get("DST", 0.0) + league.stream_boost_dst
    d["ReplPPG"] = d["Pos"].map(repl_ppg).fillna(0.0)

    # Availability sigma
    var_sigma = None
    if var.use_adp_sigma and "SigmaADP" in d.columns:
        var_sigma = pd.to_numeric(d["SigmaADP"], errors="coerce")
    if var_sigma is None or (isinstance(var_sigma, pd.Series) and var_sigma.isna().all()):
        if var.use_linear_sigma:
            var_sigma = np.clip(var.sigma_a + var.sigma_b * d["ADP"].values, var.sigma_min, var.sigma_max)
        else:
            var_sigma = np.full(len(d), var.sigma_const)
    d["Sigma"] = var_sigma

    # Baseline EV (season) above replacement
    start_share = DEFAULT_START_SHARE if league.use_predraft_startshare else {p: 1.0 for p in DEFAULT_START_SHARE}
    d["PreDraftStartShare"] = d["Pos"].map(start_share).fillna(1.0)
    d["PreWeeklyEV_base"] = np.maximum(0.0, d["PerGame"] - d["ReplPPG"]) * d["ProjG"] * d["PreDraftStartShare"]
    d["WeeklyEV"] = d["PreWeeklyEV_base"]  # back-compat alias

    # Stable PID
    d["PID"] = d.apply(lambda r: f"{r['Player']}|{r.get('Team','')}|{r['Pos']}", axis=1)

    return d

def pick_number(round_num: int, teams: int, slot: int) -> int:
    r = int(round_num); t = int(teams); s = int(slot)
    if r % 2 == 1:
        return (r - 1) * t + s
    else:
        return r * t - (s - 1)

def tail_prob(pick: int, adp: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    z = (pick - adp) / np.maximum(sigma, 1e-6)
    return 1.0 - normal_cdf(z)

def expand_window_until(pool: pd.DataFrame, adp_col: str, center_pick: int, want: int, base_window: int = 12, max_window: int = 80) -> pd.DataFrame:
    pool = pool.copy()
    pool[adp_col] = pd.to_numeric(pool[adp_col], errors="coerce")
    w = base_window
    out = pool[np.abs(pool[adp_col] - center_pick) <= w]
    while len(out) < want and w < max_window:
        w += 6
        out = pool[np.abs(pool[adp_col] - center_pick) <= w]
    return out if len(out) > 0 else pool

def include_fallers(pool: pd.DataFrame, full_df: pd.DataFrame, pick_now: int, max_fallers: int = 12) -> pd.DataFrame:
    fallers = full_df[full_df["ADP"] <= (pick_now - 1)].copy()
    if fallers.empty:
        return pool
    fallers = fallers.sort_values(["PreWeeklyEV_base", "ADP"], ascending=[False, True]).head(max_fallers)
    cat = pd.concat([pool, fallers], ignore_index=True).drop_duplicates("PID")
    return cat

def compute_targets(model_df: pd.DataFrame, round_num: int, league: LeagueConfig, var: VarianceModel, top_n: int = 12) -> pd.DataFrame:
    if model_df is None or model_df.empty:
        return pd.DataFrame()

    pick_now = pick_number(round_num, league.teams, league.slot)
    next_round = min(league.rounds, round_num + 1)
    pick_next = pick_number(next_round, league.teams, league.slot)

    df = model_df.copy()
    taken = set(st.session_state.get("taken", set()))
    mine = set(st.session_state.get("mine", set()))
    unavailable = taken.union(mine)
    df = df[~df["PID"].isin(unavailable)].copy()

    pool = expand_window_until(df, "ADP", pick_now, want=max(top_n*2, 30), base_window=12, max_window=80)
    pool = include_fallers(pool, df, pick_now, max_fallers=12)

    pool["ValueNow"] = pool["PreWeeklyEV_base"].astype(float)
    pool["PAvailNext"] = tail_prob(pick_next, pool["ADP"].values, pool["Sigma"].values).clip(0.0, 1.0)

    alt_vals: Dict[str, float] = {}
    for pos in ["QB", "RB", "WR", "TE", "K", "DST"]:
        near_next = expand_window_until(df[df["Pos"] == pos], "ADP", pick_next, want=max(10, top_n), base_window=12, max_window=80)
        alt_vals[pos] = float(near_next["PreWeeklyEV_base"].quantile(0.85)) if not near_next.empty else 0.0

    drop = np.maximum(0.0, pool["ValueNow"].values - pool["Pos"].map(alt_vals).values)
    regret = (1.0 - pool["PAvailNext"].values) * drop

    season_sd = pool["WeeklySD"].values * np.sqrt(np.maximum(pool["ProjG"].values, 1.0))
    lam = float(st.session_state.get("risk_lambda", 0.15))
    pool["RiskAdj"] = pool["ValueNow"].values - lam * season_sd + 0.5 * regret

    pool["RoundNow"] = round_num
    pool["PickNow"] = pick_now
    pool["NextPick"] = pick_next

    pool = pool.sort_values(["RiskAdj", "ValueNow", "ADP"], ascending=[False, False, True]).reset_index(drop=True)
    return pool.head(top_n)

# =============================================================================
# UI
# =============================================================================

st.set_page_config(page_title="Fantasy Draft Console", layout="centered")
st.title("Draft Console")

# Sidebar
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
    st.subheader("Upload data (required)")
    qb_up = st.file_uploader("QB projections (FantasyPros CSV)", type="csv", key="upQB")
    flx_up = st.file_uploader("FLX projections (RB/WR/TE)", type="csv", key="upFLX")
    k_up = st.file_uploader("Kicker projections", type="csv", key="upK")
    dst_up = st.file_uploader("DST projections", type="csv", key="upDST")
    adp_up = st.file_uploader("Overall ADP (CSV)", type="csv", key="upADP")

# Config objects
league = LeagueConfig(
    teams=int(teams), slot=int(slot), rounds=int(rounds),
    starters={"QB": int(s_qb), "RB": int(s_rb), "WR": int(s_wr), "TE": int(s_te), "FLEX": int(s_fx), "K": int(s_k), "DST": int(s_dst)},
)
var = VarianceModel(
    use_adp_sigma=use_adp_sigma, use_linear_sigma=use_linear_sigma,
    sigma_const=sigma_const, sigma_a=sigma_a, sigma_b=sigma_b, sigma_min=sigma_min, sigma_max=sigma_max,
)
inj = InjuryModel()
risk_by_pos = {"QB": 0.45, "RB": 0.55, "WR": 0.50, "TE": 0.50, "K": 0.40, "DST": 0.30}
cv_by_pos = {"QB": 0.35, "RB": 0.55, "WR": 0.60, "TE": 0.60, "K": 0.30, "DST": 0.25}

# Load uploads
proj_df = load_fp_uploads(qb_up, flx_up, k_up, dst_up)
adp_df = load_adp_upload(adp_up)

if proj_df is None or proj_df.empty:
    st.info("📥 Upload at least the FLX projections (RB/WR/TE). QB/K/DST are optional but recommended.")
    st.stop()

# Build model dataframe
model_df = build_model_df(proj_df, league, var, inj, risk_by_pos, cv_by_pos, adp_df=adp_df)
if model_df is None or model_df.empty:
    st.error("Model data failed to build. Check your uploads.")
    st.stop()

# Session state
if "taken" not in st.session_state: st.session_state["taken"] = set()
if "mine" not in st.session_state: st.session_state["mine"] = set()
if "history" not in st.session_state: st.session_state["history"] = []
if "current_round" not in st.session_state: st.session_state["current_round"] = 1

# Header
round_now = int(st.session_state["current_round"])
pick_now = pick_number(round_now, league.teams, league.slot)
next_round = min(league.rounds, round_now + 1)
pick_next = pick_number(next_round, league.teams, league.slot)

c1, c2, c3 = st.columns([1,1,2])
with c1: st.metric("Current round", round_now)
with c2: st.metric("Your pick #", pick_now)
with c3: st.markdown(f"**Next pick:** Round {next_round}, Overall #{pick_next}")

# Slider
topN = int(st.slider("How many recommendations to show", 5, 30, 10, 1))

# Compute targets
targets = compute_targets(model_df, round_now, league, var, top_n=topN)

# Render
st.markdown("### Recommended now")
total_avail = int((~model_df["PID"].isin(st.session_state["taken"].union(st.session_state["mine"]))).sum())
st.caption(f"Showing **{len(targets)}** of **{total_avail}** available players.")

for _, row in targets.iterrows():
    p = row["Player"]; pos = row["Pos"]; team = row.get("Team",""); pid = row["PID"]
    adp = float(row["ADP"]); pav = float(row["PAvailNext"]); val = float(row["ValueNow"])
    ra = float(row["RiskAdj"]); pg = float(row["PerGame"])

    b1, b2, b3, b4 = st.columns([5, 1.5, 1.5, 1.2])
    with b1:
        st.write(f"**{p}** — {pos} {team} | ADP {adp:.1f} | P@Next {pav:.2f} | ΔPtsNow(season) {val:.1f} | RiskScore {ra:.1f} | Pts/G {pg:.1f}")
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

# Roster & controls
st.divider()
st.markdown("#### My picks this draft")
mine_df = model_df[model_df["PID"].isin(st.session_state["mine"])][["Player","Pos","Team","PerGame","PreWeeklyEV_base"]]
if mine_df.empty:
    st.caption("No picks yet.")
else:
    mine_df = mine_df.sort_values(["Pos","PreWeeklyEV_base"], ascending=[True, False])
    st.dataframe(mine_df.rename(columns={"PreWeeklyEV_base":"ΔPtsNow(season)"}), hide_index=True, use_container_width=True)

cA, cB = st.columns([1,1])
with cA:
    if st.button("↩︎ Undo last action", type="secondary"):
        if st.session_state["history"]:
            op, pid = st.session_state["history"].pop()
            if op == "draft":
                st.session_state["mine"].discard(pid)
                st.session_state["taken"].discard(pid)
                st.session_state["current_round"] = max(1, round_now - 1)
            elif op in ("taken","hide"):
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
- **ADP** — Average Draft Position from your ADP upload (or inferred from projections order).
- **P@Next** — Probability the player is *still available at your next pick* (snake‑aware).
- **ΔPtsNow (season)** — Expected **season** points **above replacement** if you draft the player now.
- **RiskScore** — Ranking score = ΔPtsNow − λ·SeasonSD + 0.5·Regret.
- **Pts/G** — Projected points per game.
""")


