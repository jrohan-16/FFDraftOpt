# -*- coding: utf-8 -*-
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
# Helpers
# =============================================================================

def normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.erf(x / np.sqrt(2.0)))


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


def pick_number(round_num: int, teams: int, slot: int) -> int:
    if round_num % 2 == 1:  # odd
        return (round_num - 1) * teams + slot
    else:
        return round_num * teams - slot + 1


def find_fpts_col(df: pd.DataFrame) -> Optional[str]:
    names = [c for c in df.columns if "fpts" in c.lower() or "fantasy points" in c.lower() or c.lower() == "pts"]
    if not names:
        return None
    # Prefer season totals (avoid "/G")
    names = sorted(names, key=lambda c: ("/g" in c.lower(), len(c)))
    return names[0]


def find_team_col(df: pd.DataFrame) -> Optional[str]:
    for cand in ["team", "tm", "teams", "nfl team", "nfl_team", "nflteam"]:
        if cand in df.columns:
            return cand
    return None


# =============================================================================
# CSV ingestion (uploads only; no disk I/O)
# =============================================================================

@st.cache_data(show_spinner=False)
def _load_fp_from_bytes(content: bytes, pos_hint: Optional[str]) -> pd.DataFrame:
    """Internal cached loader from raw CSV bytes; normalizes to (Player, Pos, Team, Bye, ProjPts)."""
    df = pd.read_csv(io.BytesIO(content))
    df = cols_lower(df)

    # Player
    player_col = first_present(df.columns, ["player", "name", "player name"])
    if player_col is None:
        # fallback to the first column, but ensure it's stringy
        player_col = list(df.columns)[0]

    # Team & Bye
    team_col = find_team_col(df) or ""
    bye_col = first_present(df.columns, ["bye", "bye week", "bye week number"])

    # Fantasy points
    fpts_col = find_fpts_col(df)
    if fpts_col is None:
        df["__projpts__"] = 0.0
    else:
        df["__projpts__"] = pd.to_numeric(df[fpts_col], errors="coerce").fillna(0.0)

    # Position
    if pos_hint is None:
        pos_col = first_present(df.columns, ["pos", "position"])
        if pos_col is None:
            raise ValueError("Could not infer 'Pos' from the uploaded file. Please upload the correct CSV.")
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

    # Optional ADP/ECR columns if present (not required)
    for adp_col in ["adp", "avg", "ecr"]:
        if adp_col in df.columns:
            out[adp_col.upper()] = pd.to_numeric(df[adp_col], errors="coerce")

    return out


def load_fp_uploads(qb_file, flx_file, k_file, dst_file) -> pd.DataFrame:
    """
    Load FantasyPros projection uploads (some may be None). At least one must be present.
    pos_hint is used for QB/K/DST when those files lack a 'Pos' column.
    """
    frames: List[pd.DataFrame] = []
    if qb_file is not None:
        frames.append(_load_fp_from_bytes(qb_file.getvalue(), pos_hint="QB"))
    if flx_file is not None:
        frames.append(_load_fp_from_bytes(flx_file.getvalue(), pos_hint=None))  # FLX has Pos inside
    if k_file is not None:
        frames.append(_load_fp_from_bytes(k_file.getvalue(), pos_hint="K"))
    if dst_file is not None:
        frames.append(_load_fp_from_bytes(dst_file.getvalue(), pos_hint="DST"))

    if not frames:
        st.error("Please upload at least one projections CSV (QB, FLX, K, or DST).")
        st.stop()

    df_all = pd.concat(frames, ignore_index=True, axis=0)
    # De-dup by (Player, Pos) keeping the higher ProjPts (safer for multi-source merges)
    df_all = df_all.sort_values("ProjPts", ascending=False).drop_duplicates(["Player", "Pos"], keep="first").reset_index(drop=True)

    # Promote ADP if included in projections
    if "ADP" not in df_all.columns:
        if "AVG" in df_all.columns:
            df_all["ADP"] = df_all["AVG"]
        elif "ECR" in df_all.columns:
            df_all["ADP"] = df_all["ECR"]

    return df_all


@st.cache_data(show_spinner=False)
def _parse_adp_from_bytes(content: bytes) -> pd.DataFrame:
    """Parse an ADP CSV (optional). We tolerate common FantasyPros header variants."""
    df = pd.read_csv(io.BytesIO(content))
    df = cols_lower(df)

    # Player
    player_col = first_present(df.columns, ["player", "name", "player name", "player team (pos)"])
    if player_col is None:
        player_col = list(df.columns)[0]

    # If POS is in a dedicated column, use it; else try to extract from "(POS)"
    pos_col = first_present(df.columns, ["pos", "position"])
    if pos_col is not None:
        pos_series = df[pos_col].astype(str).str.upper().str.replace(" ", "", regex=False).replace({"DEF": "DST"})
    else:
        tmp = df[player_col].astype(str)
        pos_series = None
        if tmp.str.contains(r"\(", regex=True).any():
            pos_series = tmp.str.extract(r"\(([^)]+)\)")[0].str.upper().str.replace(" ", "", regex=False).replace({"DEF": "DST"})

    # ADP number
    adp_col = first_present(df.columns, ["adp", "avg", "average", "overall"])
    if adp_col is None:
        raise ValueError("ADP upload missing a recognizable ADP/AVG/Overall column.")

    # Optional dispersion
    sd_col = first_present(df.columns, ["std dev", "stdev", "stddev", "sd", "std"])
    min_col = first_present(df.columns, ["min pick", "best", "best pick", "min"])
    max_col = first_present(df.columns, ["max pick", "worst", "worst pick", "max"])

    out = pd.DataFrame(
        {
            "Player": df[player_col].astype(str).str.strip(),
            "ADP": pd.to_numeric(df[adp_col], errors="coerce"),
        }
    )
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
# Modeling
# =============================================================================

@dataclass
class LeagueConfig:
    teams: int = 14
    slot: int = 3
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
    use_linear_sigma: bool = True
    sigma_const: float = 12.0
    sigma_a: float = 3.0
    sigma_b: float = 0.04
    sigma_min: float = 6.0
    sigma_max: float = 22.0
    window: int = 12
    use_logistic: bool = False
    logistic_scale_from_sigma: bool = True
    logistic_scale_const: float = 7.0
    use_adp_sigma: bool = True


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
    start_share: Dict[str, float],
    adp_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    d = df.copy()

    # Risk default if missing
    if "Risk" not in d.columns:
        d["Risk"] = d["Pos"].map(risk_by_pos).fillna(0.5)
    d["Risk"] = pd.to_numeric(d["Risk"], errors="coerce").fillna(0.5).clip(0.05, 0.95)

    # ADP merge preference
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

    # Expected games
    pos_arr = d["Pos"].values.astype(str)
    d["ProjG"] = markov_proj_g(d["Risk"].values.astype(float), pos_arr, inj)

    # PerGame & WeeklySD
    d["PerGame"] = d["ProjPts"] / d["ProjG"].replace(0, np.nan)
    d["PerGame"] = d["PerGame"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    d["WeeklySD"] = d["PerGame"] * d["Pos"].map(cv_by_pos).fillna(0.5)

    # Replacement levels
    starters = dict(league.starters or {})
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

    repl_pts: Dict[str, float] = {}
    for p in ["QB", "RB", "WR", "TE", "K", "DST"]:
        subset = d[d["Pos"] == p].copy()
        if subset.empty:
            repl_pts[p] = 0.0
            continue
        subset["AdjPts"] = subset["ProjPts"] * (subset["ProjG"] / 17.0)
        subset = subset.sort_values("AdjPts", ascending=False)
        brank = int(max(1, min(len(subset), round(league_starters.get(p, 0)))))
        repl_pts[p] = float(subset.iloc[brank - 1]["AdjPts"])

    # Streaming boosts
    repl_pts["K"] = repl_pts.get("K", 0.0) + 17.0 * league.stream_boost_k
    repl_pts["DST"] = repl_pts.get("DST", 0.0) + 17.0 * league.stream_boost_dst

    d["AdjPts"] = d["ProjPts"] * (d["ProjG"] / 17.0)
    d["ReplAdjPts"] = d["Pos"].map(repl_pts).fillna(0.0)
    d["VBD"] = (d["AdjPts"] - d["ReplAdjPts"]).clip(lower=0.0)

    # Weekly replacement PPG
    repl_ppg: Dict[str, float] = {}
    for p in ["QB", "RB", "WR", "TE", "K", "DST"]:
        subset = d[d["Pos"] == p].copy()
        if subset.empty:
            repl_ppg[p] = 0.0
            continue
        subset = subset.sort_values("PerGame", ascending=False)
        brank = int(max(1, min(len(subset), round(league_starters.get(p, 0)))))
        repl_ppg[p] = float(subset.iloc[brank - 1]["PerGame"])

    repl_ppg["K"] = repl_ppg.get("K", 0.0) + league.stream_boost_k
    repl_ppg["DST"] = repl_ppg.get("DST", 0.0) + league.stream_boost_dst
    d["ReplPPG"] = d["Pos"].map(repl_ppg).fillna(0.0)

    # Sigma & logistic scale
    if var.use_adp_sigma and "SigmaADP" in d.columns:
        d["Sigma"] = pd.to_numeric(d["SigmaADP"], errors="coerce").clip(var.sigma_min, var.sigma_max)
    elif var.use_linear_sigma:
        d["Sigma"] = np.clip(var.sigma_a + var.sigma_b * d["ADP"].values, var.sigma_min, var.sigma_max)
    else:
        d["Sigma"] = var.sigma_const

    if var.use_logistic:
        if var.logistic_scale_from_sigma:
            d["LogisticScale"] = d["Sigma"] * (np.sqrt(3.0) / np.pi)
        else:
            d["LogisticScale"] = var.logistic_scale_const

    # Pre-draft EV baseline
    d["PreDraftStartShare"] = d["Pos"].map(start_share).fillna(1.0)
    d["PreWeeklyEV_base"] = np.maximum(0.0, d["PerGame"] - d["ReplPPG"]) * d["ProjG"] * d["PreDraftStartShare"]

    # PosRank & crude tiers
    d["PosRank"] = d.groupby("Pos")["VBD"].rank(ascending=False, method="min")

    def _tierer(sr: pd.Series) -> pd.Series:
        n = len(sr)
        ranks = sr.rank(ascending=True, method="min")
        t1, t2, t3 = 0.15 * n, 0.35 * n, 0.65 * n
        return 1 + (ranks > t1).astype(int) + (ranks > t2).astype(int) + (ranks > t3).astype(int)

    d["Tier"] = d.groupby("Pos")["PosRank"].transform(_tierer)
    return d


def tail_prob(pick: int, adp: np.ndarray, model_df: pd.DataFrame, var: VarianceModel) -> np.ndarray:
    if var.use_logistic and "LogisticScale" in model_df.columns:
        return 1.0 / (1.0 + np.exp((pick - adp) / model_df["LogisticScale"].values))
    else:
        return 1.0 - normal_cdf((pick - adp) / model_df["Sigma"].values)


def lineup_ppw_given(players: pd.DataFrame, league: LeagueConfig) -> float:
    if players.empty:
        return 0.0
    pos = players["Pos"].values
    ppg = players["PerGame"].values

    def top_sum(mask, n):
        vals = sorted(ppg[mask], reverse=True)
        return sum(vals[: max(0, n)]), vals[max(0, n) :]

    qb_s, _ = top_sum(pos == "QB", league.starters.get("QB", 0))
    rb_s, rb_r = top_sum(pos == "RB", league.starters.get("RB", 0))
    wr_s, wr_r = top_sum(pos == "WR", league.starters.get("WR", 0))
    te_s, te_r = top_sum(pos == "TE", league.starters.get("TE", 0))
    k_s, _ = top_sum(pos == "K", league.starters.get("K", 0))
    d_s, _ = top_sum(pos == "DST", league.starters.get("DST", 0))
    flex_pool = sorted(rb_r + wr_r + te_r, reverse=True)
    fx_s = sum(flex_pool[: league.starters.get("FLEX", 0)])
    return qb_s + rb_s + wr_s + te_s + k_s + d_s + fx_s


def marginal_lineup_gain(model_df: pd.DataFrame, league: LeagueConfig, candidate_row: pd.Series, current_plus: Optional[pd.DataFrame] = None) -> float:
    mine = model_df[model_df.get("Mine", 0) == 1][["Pos", "PerGame"]].copy()
    base_ppw = lineup_ppw_given(mine, league) if current_plus is None else lineup_ppw_given(current_plus, league)
    add_df = pd.DataFrame([{"Pos": candidate_row["Pos"], "PerGame": candidate_row["PerGame"]}])
    mine_plus = pd.concat([mine if current_plus is None else current_plus, add_df], ignore_index=True)
    new_ppw = lineup_ppw_given(mine_plus, league)
    delta_ppw = max(0.0, new_ppw - base_ppw)
    return delta_ppw * float(candidate_row["ProjG"])


def compute_targets(model_df: pd.DataFrame, round_num: int, league: LeagueConfig, var: VarianceModel, use_marginal: bool = False) -> pd.DataFrame:
    pick = pick_number(round_num, league.teams, league.slot)
    df = model_df.copy()
    if "Taken" in df.columns:
        df = df[df["Taken"] == 0]

    df = df[np.abs(df["ADP"] - pick) <= var.window].copy()
    pav = tail_prob(pick, df["ADP"].values, df, var)
    df["PAvail"] = pav
    df["EV_Season"] = df["VBD"].values * pav

    if not use_marginal:
        df["WeeklyEV"] = df["PreWeeklyEV_base"].values * pav
    else:
        df["WeeklyEV"] = [marginal_lineup_gain(model_df, league, row) * pav[i] for i, (_, row) in enumerate(df.iterrows())]

    cols = ["Player","Pos","Team","PerGame","VBD","ADP","ProjPts","ProjG","PAvail","EV_Season","ReplPPG","WeeklyEV","Tier","PosRank"]
    df = df.sort_values(["WeeklyEV", "Tier", "ADP"], ascending=[False, True, True])
    return df[[c for c in cols if c in df.columns]]


def snake_sim(model_df: pd.DataFrame, league: LeagueConfig, var: VarianceModel, rounds_sim: int = 8, runs: int = 200) -> pd.DataFrame:
    df = model_df.copy().sort_values("ADP").reset_index(drop=True)
    N = len(df)
    res: List[Dict[str, float]] = []
    for _ in range(int(runs)):
        if var.use_adp_sigma and "Sigma" in df.columns:
            sigma = df["Sigma"].values
        elif var.use_linear_sigma:
            sigma = np.clip(var.sigma_a + var.sigma_b * df["ADP"].values, var.sigma_min, var.sigma_max)
        else:
            sigma = np.full(N, var.sigma_const)

        draw = df["ADP"].values + np.random.normal(0, sigma)
        sim_rank = np.argsort(np.argsort(draw))

        taken = np.zeros(N, dtype=bool)
        my_picks: List[int] = []

        for r in range(1, int(rounds_sim) + 1):
            pick = pick_number(r, league.teams, league.slot) - 1
            taken[sim_rank < pick] = True
            avail_idx = np.where(~taken)[0]
            if avail_idx.size == 0:
                break
            best_idx = avail_idx[np.argmax(df.iloc[avail_idx]["PreWeeklyEV_base"].values)]
            taken[best_idx] = True
            my_picks.append(best_idx)

        sum_wev = float(df.iloc[my_picks]["PreWeeklyEV_base"].sum()) if my_picks else 0.0
        res.append({"sum_wev": sum_wev, "num_picks": float(len(my_picks))})
    return pd.DataFrame(res)


# =============================================================================
# UI
# =============================================================================

st.set_page_config(page_title="Fantasy Draft Optimizer (Streamlit MVP v2.1)", layout="wide")
st.title("Fantasy Draft Optimizer — Streamlit MVP v2.1")

with st.sidebar:
    st.header("League Settings")
    teams = st.number_input("Teams", 8, 20, 14, 1)
    slot = st.number_input("Your Draft Slot", 1, teams, 3, 1)
    rounds = st.number_input("Rounds", 10, 25, 16, 1)

    st.markdown("**Starters**")
    s_qb = st.number_input("QB", 0, 2, 1, 1)
    s_rb = st.number_input("RB", 0, 4, 2, 1)
    s_wr = st.number_input("WR", 0, 4, 2, 1)
    s_te = st.number_input("TE", 0, 3, 1, 1)
    s_flex = st.number_input("FLEX (RB/WR/TE)", 0, 3, 1, 1)
    s_k = st.number_input("K", 0, 2, 1, 1)
    s_dst = st.number_input("DST", 0, 2, 1, 1)
    bench = st.number_input("Bench Size", 0, 10, 7, 1)

    st.markdown("**FLEX shares**")
    flex_rb = st.slider("FLEX share RB", 0.0, 1.0, 0.45, 0.05)
    flex_wr = st.slider("FLEX share WR", 0.0, 1.0, 0.50, 0.05)
    flex_te = st.slider("FLEX share TE", 0.0, 1.0, 0.05, 0.05)

    st.markdown("**Streaming boost (PPG)**")
    stream_k = st.slider("K streaming boost (PPG)", 0.0, 5.0, 0.0, 0.25)
    stream_dst = st.slider("DST streaming boost (PPG)", 0.0, 5.0, 0.0, 0.25)

    use_predraft_ss = st.checkbox("Use Pre-draft StartShare in EV", True)

with st.sidebar:
    st.header("Variance / Availability model")
    use_logistic = st.checkbox("Use Logistic instead of Normal", False)
    use_adp_sigma = st.checkbox("Prefer Sigma from ADP file (if present)", True)
    use_linear_sigma = st.checkbox("If no ADP σ, use linear σ = a + b*ADP (else constant)", True)
    sigma_const = st.number_input("σ (constant)", 1.0, 40.0, 12.0, 0.5)
    sigma_a = st.number_input("σ = a + b*ADP (a)", 0.0, 15.0, 3.0, 0.5)
    sigma_b = st.number_input("σ = a + b*ADP (b)", 0.0, 0.50, 0.04, 0.005)
    sigma_min = st.number_input("σ min", 1.0, 40.0, 6.0, 0.5)
    sigma_max = st.number_input("σ max", 1.0, 60.0, 22.0, 0.5)
    window = st.number_input("Candidate window (± picks)", 2, 60, 12, 1)
    log_scale_const = st.number_input("Logistic scale (if not from σ)", 1.0, 20.0, 7.0, 0.5)

with st.sidebar:
    st.header("Risk & Weekly Volatility")
    r_qb = st.slider("QB base risk", 0.0, 1.0, 0.35, 0.05)
    r_rb = st.slider("RB base risk", 0.0, 1.0, 0.55, 0.05)
    r_wr = st.slider("WR base risk", 0.0, 1.0, 0.50, 0.05)
    r_te = st.slider("TE base risk", 0.0, 1.0, 0.45, 0.05)
    r_k = st.slider("K base risk", 0.0, 1.0, 0.20, 0.05)
    r_dst = st.slider("DST base risk", 0.0, 1.0, 0.10, 0.05)

    cv_qb = st.slider("QB CV (weekly SD / PG)", 0.0, 1.0, 0.35, 0.05)
    cv_rb = st.slider("RB CV", 0.0, 1.0, 0.55, 0.05)
    cv_wr = st.slider("WR CV", 0.0, 1.0, 0.60, 0.05)
    cv_te = st.slider("TE CV", 0.0, 1.0, 0.60, 0.05)
    cv_k = st.slider("K CV", 0.0, 1.0, 0.30, 0.05)
    cv_dst = st.slider("DST CV", 0.0, 1.0, 0.25, 0.05)

with st.sidebar:
    st.header("File uploads (no defaults)")
    up_qb = st.file_uploader("Projections — QB (CSV)", type=["csv"])
    up_flx = st.file_uploader("Projections — FLX (RB/WR/TE) (CSV)", type=["csv"])
    up_k = st.file_uploader("Projections — K (CSV)", type=["csv"])
    up_dst = st.file_uploader("Projections — DST/DEF (CSV)", type=["csv"])
    up_adp = st.file_uploader("ADP (CSV, optional)", type=["csv"])
    st.caption("Tip: Upload any subset. ADP is optional. Nothing is saved to disk.")

# Prepare configs
league = LeagueConfig(
    teams=int(teams),
    slot=int(slot),
    rounds=int(rounds),
    starters={"QB": int(s_qb), "RB": int(s_rb), "WR": int(s_wr), "TE": int(s_te), "FLEX": int(s_flex), "K": int(s_k), "DST": int(s_dst)},
    bench=int(bench),
    flex_shares={"RB": float(flex_rb), "WR": float(flex_wr), "TE": float(flex_te)},
    use_predraft_startshare=True if use_predraft_ss else False,
    stream_boost_k=float(stream_k),
    stream_boost_dst=float(stream_dst),
)

var = VarianceModel(
    use_linear_sigma=bool(use_linear_sigma),
    sigma_const=float(sigma_const),
    sigma_a=float(sigma_a),
    sigma_b=float(sigma_b),
    sigma_min=float(sigma_min),
    sigma_max=float(sigma_max),
    window=int(window),
    use_logistic=bool(use_logistic),
    logistic_scale_from_sigma=True,
    logistic_scale_const=float(log_scale_const),
    use_adp_sigma=bool(use_adp_sigma),
)

inj = InjuryModel()

risk_by_pos = {"QB": float(r_qb), "RB": float(r_rb), "WR": float(r_wr), "TE": float(r_te), "K": float(r_k), "DST": float(r_dst)}
cv_by_pos = {"QB": 0.35, "RB": 0.55, "WR": 0.60, "TE": 0.60, "K": 0.30, "DST": 0.25}
cv_by_pos.update({"QB": min(max(cv_by_pos["QB"], 0.0), 1.0)})
cv_by_pos.update({"RB": min(max(cv_by_pos["RB"], 0.0), 1.0)})
cv_by_pos.update({"WR": min(max(cv_by_pos["WR"], 0.0), 1.0)})
cv_by_pos.update({"TE": min(max(cv_by_pos["TE"], 0.0), 1.0)})
cv_by_pos.update({"K":  min(max(cv_by_pos["K"],  0.0), 1.0)})
cv_by_pos.update({"DST":min(max(cv_by_pos["DST"],0.0), 1.0)})

start_share = DEFAULT_START_SHARE if league.use_predraft_startshare else {p: 1.0 for p in DEFAULT_START_SHARE}

# Session state for flags
if "taken" not in st.session_state:
    st.session_state["taken"] = set()
if "mine" not in st.session_state:
    st.session_state["mine"] = set()

# Load projections (uploads only)
df_all = load_fp_uploads(up_qb, up_flx, up_k, up_dst)

# Optional ADP (uploads only)
adp_df = None
if up_adp is not None:
    try:
        adp_df = _parse_adp_from_bytes(up_adp.getvalue())
    except Exception as e:
        st.warning(f"Could not parse uploaded ADP: {e}")
        adp_df = None

# Build model
model_df = build_model_df(df_all, league, var, inj, risk_by_pos, cv_by_pos, start_share, adp_df=adp_df)

# Apply flags
model_df["Taken"] = model_df["Player"].isin(st.session_state["taken"]).astype(int)
model_df["Mine"] = model_df["Player"].isin(st.session_state["mine"]).astype(int)

# Overview
st.subheader("Model overview")
st.caption("Snapshot of the model table (top 20 by ProjPts).")
st.dataframe(model_df.sort_values("ProjPts", ascending=False).head(20))

# On the clock
st.subheader("On the clock")
round_num = st.number_input("Round", 1, int(rounds), 1, 1)
use_marginal = st.checkbox("Rank by roster-based marginal WeeklyEV (Δ lineup PPW)", True)
targets_df = compute_targets(model_df, int(round_num), league, var, use_marginal=use_marginal)
st.dataframe(targets_df.head(60))

with st.expander("Mark picks (Taken/Mine)"):
    taken_add = st.multiselect("Mark TAKEN", list(targets_df["Player"].values), key="taken_ms")
    mine_add = st.multiselect("Mark MINE", list(targets_df["Player"].values), key="mine_ms")
    colA, colB, colC = st.columns(3)
    with colA:
        apply = st.button("Apply flags", key="apply_flags")
    with colB:
        reset_taken = st.button("Reset TAKEN", key="reset_taken")
    with colC:
        reset_mine = st.button("Reset MINE", key="reset_mine")
    if apply:
        st.session_state["taken"].update(taken_add)
        st.session_state["mine"].update(mine_add)
        st.success("Flags applied. Rerun to update lists.")
    if reset_taken:
        st.session_state["taken"] = set()
        st.success("Taken cleared.")
    if reset_mine:
        st.session_state["mine"] = set()
        st.success("Mine cleared.")

# Calibration (no disk I/O)
st.subheader("Calibration: P(Avail) curve")
sel_player = st.selectbox("Player", options=model_df["Player"].unique().tolist()[:500])
sel_row = model_df[model_df["Player"] == sel_player].iloc[0]
if var.use_logistic and "LogisticScale" in model_df.columns:
    scale_val = float(sel_row["LogisticScale"])
    curve = [1.0 / (1.0 + math.exp((p - sel_row["ADP"]) / scale_val)) for p in range(1, league.teams * rounds + 1)]
else:
    sigma_val = float(sel_row["Sigma"])
    curve = [1.0 - float(normal_cdf((p - sel_row["ADP"]) / sigma_val)) for p in range(1, league.teams * rounds + 1)]
st.line_chart(pd.DataFrame({"Pick": list(range(1, league.teams * rounds + 1)), "PAvail": curve}).set_index("Pick"))

# SnakeSim (pure in-memory)
st.subheader("SnakeSim (others pick by ADP+noise; you by PreDraft WeeklyEV)")
colS1, colS2 = st.columns(2)
with colS1:
    rounds_sim = st.number_input("Simulate N rounds", 2, int(rounds), 8, 1)
with colS2:
    runs_sim = st.number_input("Runs", 50, 2000, 200, 50)
snake_df = snake_sim(model_df, league, var, rounds_sim=int(rounds_sim), runs=int(runs_sim))
st.write(f"Runs: {len(snake_df)} | Mean sum WeeklyEV over first {int(rounds_sim)} rounds: {snake_df['sum_wev'].mean():.1f}")
st.bar_chart(snake_df["sum_wev"])

# Exports (memory only)
st.subheader("Exports")
csv = targets_df.to_csv(index=False).encode("utf-8")
st.download_button("Download targets (CSV)", csv, file_name="targets_round.csv", mime="text/csv")

state = {"taken": list(st.session_state["taken"]), "mine": list(st.session_state["mine"])}
st.download_button("Download flags (JSON)", json.dumps(state, indent=2).encode("utf-8"), file_name="draft_state.json", mime="application/json")

st.caption("Uploads only; no defaults, no file writes. Projections can be any of QB/FLX/K/DST; ADP is optional.")

