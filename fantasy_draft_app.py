# -*- coding: utf-8 -*-
"""
Fantasy Draft Assistant — Minimal Draft Console (v3.0)

Goal: Make it *very easy* to (1) upload data, (2) mark picks as TAKEN or MINE,
and (3) see a *clear, auto-updating* short list of top recommended picks.

Design principles:
- Uploads only (no default paths, no file writes).
- Minimal UI: one console to manage picks and see recommendations.
- Keep model intact; just surface the output simply and clearly.
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# =============================================================================
# Small math helpers (no SciPy; no np.erf dependency)
# =============================================================================

def normal_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF without relying on np.erf or scipy.
    Uses the Abramowitz–Stegun 7.1.26 approximation for erf.
    """
    x = np.asarray(x, dtype=float)
    y = x / np.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * np.abs(y))
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t
    erf_y = 1.0 - poly * np.exp(-y * y)
    erf_y = np.where(y >= 0.0, erf_y, -erf_y)
    return 0.5 * (1.0 + erf_y)


def pick_number(round_num: int, teams: int, slot: int) -> int:
    """Snake draft overall pick for a given round and your slot."""
    if round_num % 2 == 1:  # odd round
        return (round_num - 1) * teams + slot
    else:
        return round_num * teams - slot + 1


def first_present(ls: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    s = {c.lower() for c in ls}
    for c in candidates:
        if c.lower() in s:
            return c
    return None


def cols_lower(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]
    lower = {c: c.lower() for c in out.columns}
    return out.rename(columns=lower)


def find_fpts_col(df: pd.DataFrame) -> Optional[str]:
    names = [c for c in df.columns if "fpts" in c.lower() or "fantasy points" in c.lower() or c.lower() == "pts"]
    if not names:
        return None
    names = sorted(names, key=lambda c: ("/g" in c.lower(), len(c)))  # prefer season total-like
    return names[0]


def find_team_col(df: pd.DataFrame) -> Optional[str]:
    for cand in ["team", "tm", "teams", "nfl team", "nfl_team", "nflteam"]:
        if cand in df.columns:
            return cand
    return None


# =============================================================================
# Upload ingestion (in-memory only)
# =============================================================================

@st.cache_data(show_spinner=False)
def _load_fp_from_bytes(content: bytes, pos_hint: Optional[str]) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(content))
    df = cols_lower(df)

    player_col = first_present(df.columns, ["player", "name", "player name"]) or list(df.columns)[0]
    team_col = find_team_col(df) or ""
    bye_col = first_present(df.columns, ["bye", "bye week", "bye week number"])

    fpts_col = find_fpts_col(df)
    if fpts_col is None:
        df["__projpts__"] = 0.0
    else:
        df["__projpts__"] = pd.to_numeric(df[fpts_col], errors="coerce").fillna(0.0)

    if pos_hint is None:
        pos_col = first_present(df.columns, ["pos", "position"])
        if pos_col is None:
            raise ValueError("Could not infer 'Pos' from FLX file. Please ensure it has a Pos column.")
        pos_series = df[pos_col].astype(str).str.upper().str.replace(" ", "", regex=False).replace({"DEF": "DST"})
    else:
        pos_series = pos_hint

    out = pd.DataFrame(
        {
            "Player": df[player_col].astype(str).str.strip(),
            "Pos": pos_series if isinstance(pos_series, str) else pos_series.astype(str),
            "Team": df[team_col].astype(str).str.upper().str.strip() if team_col in df.columns else "",
            "Bye": df[bye_col] if (bye_col is not None and bye_col in df.columns) else np.nan,
            "ProjPts": df["__projpts__"].astype(float),
        }
    )

    # Optional ADP/ECR
    for adp_col in ["adp", "avg", "ecr"]:
        if adp_col in df.columns:
            out[adp_col.upper()] = pd.to_numeric(df[adp_col], errors="coerce")

    return out


def load_fp_uploads(qb_file, flx_file, k_file, dst_file) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    if qb_file is not None:
        frames.append(_load_fp_from_bytes(qb_file.getvalue(), pos_hint="QB"))
    if flx_file is not None:
        frames.append(_load_fp_from_bytes(flx_file.getvalue(), pos_hint=None))  # FLX carries its Pos column
    if k_file is not None:
        frames.append(_load_fp_from_bytes(k_file.getvalue(), pos_hint="K"))
    if dst_file is not None:
        frames.append(_load_fp_from_bytes(dst_file.getvalue(), pos_hint="DST"))

    if not frames:
        st.error("Please upload at least one projections CSV (QB, FLX, K, or DST).")
        st.stop()

    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all.sort_values("ProjPts", ascending=False).drop_duplicates(["Player", "Pos"], keep="first").reset_index(drop=True)

    # If ADP isn't present from projections, it's fine (we'll backfill by projections rank if needed)
    if "ADP" not in df_all.columns:
        if "AVG" in df_all.columns:
            df_all["ADP"] = df_all["AVG"]
        elif "ECR" in df_all.columns:
            df_all["ADP"] = df_all["ECR"]

    return df_all


@st.cache_data(show_spinner=False)
def _parse_adp_from_bytes(content: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(content))
    df = cols_lower(df)

    player_col = first_present(df.columns, ["player", "name", "player name", "player team (pos)"]) or list(df.columns)[0]
    pos_col = first_present(df.columns, ["pos", "position"])
    if pos_col is not None:
        pos_series = df[pos_col].astype(str).str.upper().str.replace(" ", "", regex=False).replace({"DEF": "DST"})
    else:
        tmp = df[player_col].astype(str)
        pos_series = None
        if tmp.str.contains(r"\(", regex=True).any():
            pos_series = tmp.str.extract(r"\(([^)]+)\)")[0].str.upper().str.replace(" ", "", regex=False).replace({"DEF": "DST"})

    adp_col = first_present(df.columns, ["adp", "avg", "average", "overall"])
    if adp_col is None:
        raise ValueError("ADP upload missing ADP/AVG/Overall column.")

    sd_col = first_present(df.columns, ["std dev", "stdev", "stddev", "sd", "std"])
    min_col = first_present(df.columns, ["min pick", "best", "best pick", "min"])
    max_col = first_present(df.columns, ["max pick", "worst", "worst pick", "max"])

    out = pd.DataFrame({"Player": df[player_col].astype(str).str.strip(), "ADP": pd.to_numeric(df[adp_col], errors="coerce")})
    if pos_series is not None:
        out["Pos"] = pos_series

    sigma = None
    if sd_col is not None:
        sigma = pd.to_numeric(df[sd_col], errors="coerce")
    elif min_col is not None and max_col is not None:
        rng = pd.to_numeric(df[max_col], errors="coerce") - pd.to_numeric(df[min_col], errors="coerce")
        sigma = rng / 4.0
    if sigma is not None:
        out["SigmaADP"] = sigma
    return out


# =============================================================================
# Model (kept intact, simplified knobs)
# =============================================================================

@dataclass
class LeagueConfig:
    teams: int = 12
    slot: int = 6
    rounds: int = 16
    starters: Dict[str, int] | None = None
    bench: int = 7
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
    sigma_max: float = 22.0
    window: int = 12  # candidate window
    use_logistic: bool = False
    logistic_scale_from_sigma: bool = True
    logistic_scale_const: float = 7.0


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
    d = df.copy()

    # Risk default
    if "Risk" not in d.columns:
        d["Risk"] = d["Pos"].map(risk_by_pos).fillna(0.5)
    d["Risk"] = pd.to_numeric(d["Risk"], errors="coerce").fillna(0.5).clip(0.05, 0.95)

    # ADP merge
    if adp_df is not None:
        if "Pos" in adp_df.columns and adp_df["Pos"].notna().any():
            cols = ["Player", "Pos", "ADP"] + (["SigmaADP"] if "SigmaADP" in adp_df.columns else [])
            d = d.merge(adp_df[cols], on=["Player", "Pos"], how="left", suffixes=("", "_adpfile"))
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

    # ProjG, PerGame, WeeklySD
    pos_arr = d["Pos"].values.astype(str)
    d["ProjG"] = markov_proj_g(d["Risk"].values.astype(float), pos_arr, inj)
    d["PerGame"] = d["ProjPts"] / d["ProjG"].replace(0, np.nan)
    d["PerGame"] = d["PerGame"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    d["WeeklySD"] = d["PerGame"] * d["Pos"].map(cv_by_pos).fillna(0.5)

    # Replacement PPG (for WeeklyEV baseline); simple league starters approximation
    starters = league.starters or {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "DST": 1}
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
        subset = d[d["Pos"] == p].copy()
        if subset.empty:
            repl_ppg[p] = 0.0
            continue
        subset = subset.sort_values("PerGame", ascending=False)
        brank = int(max(1, min(len(subset), round(league_starters.get(p, 0)))))
        repl_ppg[p] = float(subset.iloc[brank - 1]["PerGame"])
    # Apply small streaming boosts if any
    repl_ppg["K"] = repl_ppg.get("K", 0.0) + league.stream_boost_k
    repl_ppg["DST"] = repl_ppg.get("DST", 0.0) + league.stream_boost_dst
    d["ReplPPG"] = d["Pos"].map(repl_ppg).fillna(0.0)

    # Availability sigma & logistic scale
    var_sigma = None
    if var.use_adp_sigma and "SigmaADP" in d.columns:
        var_sigma = pd.to_numeric(d["SigmaADP"], errors="coerce")
    if var_sigma is None or var_sigma.isna().all():
        if var.use_linear_sigma:
            var_sigma = np.clip(var.sigma_a + var.sigma_b * d["ADP"].values, var.sigma_min, var.sigma_max)
        else:
            var_sigma = np.full(len(d), var.sigma_const)
    d["Sigma"] = var_sigma
    if var.use_logistic:
        if var.logistic_scale_from_sigma:
            d["LogisticScale"] = d["Sigma"] * (np.sqrt(3.0) / np.pi)
        else:
            d["LogisticScale"] = var.logistic_scale_const

    # Baseline WeeklyEV (pre-draft start share)
    start_share = DEFAULT_START_SHARE if league.use_predraft_startshare else {p: 1.0 for p in DEFAULT_START_SHARE}
    d["PreDraftStartShare"] = d["Pos"].map(start_share).fillna(1.0)
    d["PreWeeklyEV_base"] = np.maximum(0.0, d["PerGame"] - d["ReplPPG"]) * d["ProjG"] * d["PreDraftStartShare"]
    # Back-compat alias for older UI code paths expecting "WeeklyEV"
    d["WeeklyEV"] = d["PreWeeklyEV_base"]

    return d


def tail_prob(pick: int, adp: np.ndarray, model_df: pd.DataFrame, var: VarianceModel) -> np.ndarray:
    if var.use_logistic and "LogisticScale" in model_df.columns:
        return 1.0 / (1.0 + np.exp((pick - adp) / model_df["LogisticScale"].values))
    else:
        return 1.0 - normal_cdf((pick - adp) / model_df["Sigma"].values)


def marginal_lineup_gain(model_df: pd.DataFrame, league: LeagueConfig, candidate_row: pd.Series) -> float:
    """Very minimal marginal: add candidate, see lineup PPW delta vs simple replacement lineup."""
    # Build current roster (only players marked Mine)
    mine = model_df[model_df.get("Mine", 0) == 1].copy()

    def lineup_ppw(players: pd.DataFrame) -> float:
        if players.empty:
            return 0.0
        pos = players["Pos"].values
        ppg = players["PerGame"].values
        def top_sum(mask, n):
            vals = sorted(ppg[mask], reverse=True)
            return sum(vals[: max(0, n)]), vals[max(0, n):]
        starters = league.starters or {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "DST": 1}
        qb_s, _ = top_sum(pos == "QB", starters.get("QB", 0))
        rb_s, rb_r = top_sum(pos == "RB", starters.get("RB", 0))
        wr_s, wr_r = top_sum(pos == "WR", starters.get("WR", 0))
        te_s, te_r = top_sum(pos == "TE", starters.get("TE", 0))
        k_s, _ = top_sum(pos == "K", starters.get("K", 0))
        d_s, _ = top_sum(pos == "DST", starters.get("DST", 0))
        flex_pool = sorted(rb_r + wr_r + te_r, reverse=True)
        fx_s = sum(flex_pool[: starters.get("FLEX", 0)])
        return qb_s + rb_s + wr_s + te_s + k_s + d_s + fx_s

    base_ppw = lineup_ppw(mine[["Pos", "PerGame"]])
    mine_plus = pd.concat(
        [mine[["Pos", "PerGame"]], pd.DataFrame([{"Pos": candidate_row["Pos"], "PerGame": candidate_row["PerGame"]}])],
        ignore_index=True
    )
    new_ppw = lineup_ppw(mine_plus)
    delta_ppw = max(0.0, new_ppw - base_ppw)
    return delta_ppw * float(candidate_row["ProjG"])



def compute_targets(model_df: pd.DataFrame, round_num: int, league: LeagueConfig, var: VarianceModel, use_marginal: bool = True) -> pd.DataFrame:
    """
    Improved target selection:
      - Uses *actual* value if drafted now (no availability multiplier at the current pick).
      - Incorporates probability the player is still there at your NEXT pick (snake-aware).
      - Adds a regret term: expected drop-off if you pass and the player is gone by your next pick.
      - Light position-run awareness: boosts positions likely to thin before your next pick.
    Returns a DataFrame sorted by a Risk-Adjusted score with helpful columns for transparency.
    """
    # Picks now and next (snake draft)
    pick_now = pick_number(round_num, league.teams, league.slot)
    next_round = min(league.rounds, round_num + 1)
    pick_next = pick_number(next_round, league.teams, league.slot)

    df = model_df.copy()
    if "Taken" in df.columns:
        df = df[df["Taken"] == 0].copy()

    # Candidate window for speed & relevance around *current* pick
    df = df[np.abs(df["ADP"] - pick_now) <= max(6, int(var.window))].copy()
    if df.empty:
        return pd.DataFrame(columns=["Player","Pos","Team","ADP","PAvailNext","ValueNow","RiskAdj","PerGame"])

    # Value if drafted now (roster-aware vs baseline)
    if use_marginal:
        val_now = []
        for _, row in df.iterrows():
            try:
                val_now.append(marginal_lineup_gain(model_df, league, row))
            except Exception:
                # Fail safe to baseline if marginal calc has an issue
                val_now.append(float(row.get("PreWeeklyEV_base", 0.0)))
        df["ValueNow"] = np.array(val_now, dtype=float)
    else:
        df["ValueNow"] = df.get("PreWeeklyEV_base", 0.0).astype(float)

    # Probability player is STILL available at your NEXT pick
    p_next = tail_prob(pick_next, df["ADP"].values, df, var)
    df["PAvailNext"] = p_next

    # Estimate "best alternative" value at next pick by position using baseline EV around next pick
    alt_vals = {}
    window = max(6, int(var.window))
    for pos, sub in df.groupby("Pos"):
        # Candidates near next pick from the full (not taken) model_df to avoid bias from current df window
        pool = model_df[(model_df.get("Taken", 0) == 0) & (model_df["Pos"] == pos)]
        pool = pool[np.abs(pool["ADP"] - pick_next) <= window]
        if pool.empty:
            alt_vals[pos] = 0.0
        else:
            # Use baseline EV as a robust, roster-agnostic proxy
            alt_vals[pos] = float(pool["PreWeeklyEV_base"].quantile(0.85))

    # Regret if pass (expected drop-off vs alternative at next pick)
    drop = np.maximum(0.0, df["ValueNow"].values - df["Pos"].map(alt_vals).values)
    regret = (1.0 - df["PAvailNext"].values) * drop

    # Light position-run awareness: expected # drafted before your next pick
    # Use availability model on the full board to estimate thinning by position
    full = model_df[model_df.get("Taken", 0) == 0].copy()
    p_now_full = tail_prob(pick_now, full["ADP"].values, full, var)
    p_next_full = tail_prob(pick_next, full["ADP"].values, full, var)
    drafted_between = np.maximum(0.0, p_now_full - p_next_full)  # expected removals in the gap
    full = full.assign(_gap_loss=drafted_between)
    run_intensity = full.groupby("Pos")["_gap_loss"].sum().to_dict()
    # Normalize by picks between to keep the scale stable
    picks_between = max(1, pick_next - pick_now)
    run_intensity = {k: float(v) / picks_between for k, v in run_intensity.items()}

    # Apply a mild boost proportional to run intensity at the candidate's position
    alpha = 0.20  # strength of run awareness
    pos_boost = np.array([ (1.0 + alpha * run_intensity.get(pos, 0.0)) for pos in df["Pos"].values ], dtype=float)

    # Soften K/DST earlier by modestly discounting ValueNow (streaming mindset)
    pos_soft = df["Pos"].map({"K": 0.85, "DST": 0.88}).fillna(1.0).values.astype(float)

    # Final risk-adjusted score
    risk_adj = (df["ValueNow"].values * pos_soft * pos_boost) + regret
    df["RiskAdj"] = risk_adj

    # Output columns
    keep = ["Player","Pos","Team","ADP","PAvailNext","ValueNow","RiskAdj","PerGame"]
    out = df.copy()
    out = out.sort_values(["RiskAdj","ValueNow","ADP"], ascending=[False, False, True])[keep].reset_index(drop=True)
    return out
# =============================================================================
# UI — Minimal Draft Console
# =============================================================================

st.set_page_config(page_title="Fantasy Draft Assistant — Minimal Console", layout="wide")
st.title("Fantasy Draft Assistant — Minimal Draft Console")

with st.sidebar:
    st.header("1) Upload files")
    up_qb = st.file_uploader("QB projections (CSV)", type=["csv"])
    up_flx = st.file_uploader("FLX projections (RB/WR/TE) (CSV)", type=["csv"])
    up_k = st.file_uploader("K projections (CSV)", type=["csv"])
    up_dst = st.file_uploader("DST/DEF projections (CSV)", type=["csv"])
    up_adp = st.file_uploader("ADP (CSV, optional)", type=["csv"])

    st.header("2) League")
    teams = st.number_input("Teams", 8, 20, 12, 1)
    slot = st.number_input("Your draft slot", 1, teams, 6, 1)
    rounds = st.number_input("Total rounds", 10, 25, 16, 1)
    st.caption("Starters")
    cols = st.columns(6)
    with cols[0]: s_qb = st.number_input("QB", 0, 2, 1, 1)
    with cols[1]: s_rb = st.number_input("RB", 0, 4, 2, 1)
    with cols[2]: s_wr = st.number_input("WR", 0, 4, 2, 1)
    with cols[3]: s_te = st.number_input("TE", 0, 3, 1, 1)
    with cols[4]: s_fx = st.number_input("FLEX", 0, 3, 1, 1)
    with cols[5]:
        s_k = st.number_input("K", 0, 2, 1, 1)
        s_dst = st.number_input("DST", 0, 2, 1, 1)

    with st.expander("Advanced settings", expanded=False):
        st.write("Availability model & window")
        use_adp_sigma = st.checkbox("Prefer Sigma from ADP (if present)", True)
        use_linear_sigma = st.checkbox("If no ADP σ, use linear σ = a + b·ADP (else constant)", True)
        sigma_const = st.number_input("σ (constant)", 1.0, 40.0, 12.0, 0.5)
        sigma_a = st.number_input("σ = a + b·ADP (a)", 0.0, 15.0, 3.0, 0.5)
        sigma_b = st.number_input("σ = a + b·ADP (b)", 0.0, 0.50, 0.04, 0.005)
        sigma_min = st.number_input("σ min", 1.0, 40.0, 6.0, 0.5)
        sigma_max = st.number_input("σ max", 1.0, 60.0, 22.0, 0.5)
        window = st.number_input("Candidate window (± picks)", 4, 60, 12, 1)
        st.write("Weekly volatility (CV ~ SD/PG) – reasonable defaults")
        cv_qb = st.slider("QB CV", 0.0, 1.0, 0.35, 0.05)
        cv_rb = st.slider("RB CV", 0.0, 1.0, 0.55, 0.05)
        cv_wr = st.slider("WR CV", 0.0, 1.0, 0.60, 0.05)
        cv_te = st.slider("TE CV", 0.0, 1.0, 0.60, 0.05)
        cv_k  = st.slider("K  CV", 0.0, 1.0, 0.30, 0.05)
        cv_dst= st.slider("DST CV",0.0, 1.0, 0.25, 0.05)

# Build configs
league = LeagueConfig(
    teams=int(teams), slot=int(slot), rounds=int(rounds),
    starters={"QB": int(s_qb), "RB": int(s_rb), "WR": int(s_wr), "TE": int(s_te), "FLEX": int(s_fx), "K": int(s_k), "DST": int(s_dst)},
)
var = VarianceModel(
    use_adp_sigma=bool(use_adp_sigma), use_linear_sigma=bool(use_linear_sigma),
    sigma_const=float(sigma_const), sigma_a=float(sigma_a), sigma_b=float(sigma_b),
    sigma_min=float(sigma_min), sigma_max=float(sigma_max), window=int(window),
    use_logistic=False, logistic_scale_from_sigma=True, logistic_scale_const=7.0,
)
inj = InjuryModel()

risk_by_pos = {"QB": 0.35, "RB": 0.55, "WR": 0.50, "TE": 0.45, "K": 0.20, "DST": 0.10}
cv_by_pos = {"QB": float(cv_qb), "RB": float(cv_rb), "WR": float(cv_wr), "TE": float(cv_te), "K": float(cv_k), "DST": float(cv_dst)}
for _p in list(cv_by_pos.keys()):
    cv_by_pos[_p] = min(max(cv_by_pos[_p], 0.0), 1.0)

# Load data
df_all = load_fp_uploads(up_qb, up_flx, up_k, up_dst)

adp_df = None
if up_adp is not None:
    try:
        adp_df = _parse_adp_from_bytes(up_adp.getvalue())
    except Exception as e:
        st.warning(f"ADP parse warning: {e}")

# Build model DF
model_df = build_model_df(df_all, league, var, inj, risk_by_pos, cv_by_pos, adp_df=adp_df)

# Initialize flags & round state
if "taken" not in st.session_state:
    st.session_state["taken"] = set()
if "mine" not in st.session_state:
    st.session_state["mine"] = set()
if "current_round" not in st.session_state:
    st.session_state["current_round"] = 1

# Apply flags
model_df["Taken"] = model_df["Player"].isin(st.session_state["taken"]).astype(int)
model_df["Mine"]  = model_df["Player"].isin(st.session_state["mine"]).astype(int)

# =============================================================================
# Draft Console
# =============================================================================

st.subheader("Draft Console")

col_meta1, col_meta2, col_meta3 = st.columns([1,1,2])
with col_meta1:
    st.metric("Current round", st.session_state["current_round"])
with col_meta2:
    cur_pick = pick_number(st.session_state["current_round"], league.teams, league.slot)
    st.metric("Your pick #", cur_pick)
with col_meta3:
    nxt_round = min(league.rounds, st.session_state["current_round"] + 1)
    nxt_pick = pick_number(nxt_round, league.teams, league.slot)
    st.caption(f"Next pick: Round {nxt_round}, Overall #{nxt_pick}")

# Recommended picks (top N)
use_marginal = True  # keep it simple; roster-aware by default
targets = compute_targets(model_df, int(st.session_state["current_round"]), league, var, use_marginal=use_marginal)
topN = int(st.slider("How many recommendations to show", 5, 30, 12, 1))
simple_cols = targets[["Player", "Pos", "Team", "ADP", "PAvailNext", "ValueNow", "RiskAdj", "PerGame"]].head(topN)

st.markdown("### Recommended now")
for i, row in simple_cols.iterrows():
    p = row["Player"]; pos = row["Pos"]; team = row["Team"]
    adp = row["ADP"]; pav = row["PAvailNext"]; val = row["ValueNow"]; ra = row["RiskAdj"]; pg = row["PerGame"]
    b1, b2, b3, b4 = st.columns([4, 2, 2, 2])
    with b1:
        st.write(f"**{p}** — {pos} {team} | ADP {adp:.1f} | P_next {pav:.2f} | Value {val:.1f} | RiskAdj {ra:.1f} | Pg {pg:.1f}")
    with b2:
        if st.button("Draft", key=f"mine_btn_{i}"):
            st.session_state["mine"].add(p)
            st.session_state["taken"].discard(p)
            st.session_state["current_round"] = min(league.rounds, st.session_state["current_round"] + 1)
            st.rerun()
    with b3:
        if st.button("Mark taken", key=f"taken_btn_{i}"):
            st.session_state["taken"].add(p)
            st.session_state["mine"].discard(p)
            st.rerun()
    with b4:
        if st.button("Hide", key=f"hide_btn_{i}"):
            st.session_state["taken"].add(p)  # hide acts like 'taken' for recommendation purposes
            st.rerun()

st.divider()

# Quick search & mark
st.markdown("### Quick search")
q = st.text_input("Find a player (type part of name)", "")
if q:
    sub = model_df[model_df["Player"].str.contains(q, case=False, na=False) & (model_df["Taken"] == 0)].copy()
    sub = sub.sort_values(["Mine","PreWeeklyEV_base","ADP"], ascending=[False, False, True]).head(25)
    for i, row in sub.iterrows():
        p = row["Player"]; pos = row["Pos"]; team = row["Team"]
        cols = st.columns([5,1,1])
        with cols[0]:
            st.write(f"{p} — {pos} {team}")
        with cols[1]:
            if st.button("Mine", key=f"q_m_{i}"):
                st.session_state["mine"].add(p); st.session_state["taken"].discard(p); st.rerun()
        with cols[2]:
            if st.button("Taken", key=f"q_t_{i}"):
                st.session_state["taken"].add(p); st.session_state["mine"].discard(p); st.rerun()

# Controls row
c1, c2, c3, c4 = st.columns(4)
with c1:
    if st.button("Advance to next round"):
        st.session_state["current_round"] = min(league.rounds, st.session_state["current_round"] + 1)
        st.rerun()
with c2:
    if st.button("Back one round"):
        st.session_state["current_round"] = max(1, st.session_state["current_round"] - 1)
        st.rerun()
with c3:
    if st.button("Reset flags (clear Taken/Mine)"):
        st.session_state["taken"] = set(); st.session_state["mine"] = set(); st.rerun()
with c4:
    st.download_button("Download draft state (JSON)", json.dumps({"taken": list(st.session_state["taken"]), "mine": list(st.session_state["mine"]), "round": st.session_state["current_round"]}, indent=2).encode("utf-8"), file_name="draft_state.json", mime="application/json")

st.caption("Tip: Use the 'Draft' buttons to add players to your roster and auto-advance the round. Mark others 'Taken' to keep recommendations clean.")
