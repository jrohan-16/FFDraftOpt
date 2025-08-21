# -*- coding: utf-8 -*-
"""
fantasy_draft_engine.py

Pure-Python engine for Fantasy Draft recommendations.
This module contains **no Streamlit dependencies** and can be imported by any front end.
It covers:
- CSV ingestion (from bytes)
- Modeling (injury/availability, replacement levels, EV above replacement)
- Draft pick math (snake-aware pick numbers, availability tails)
- Target recommendations (risk-adjusted, next-pick lookahead)

Ported from the original monolithic Streamlit app. See the UI app for cached wrappers.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd


# =============================================================================
# Utility math
# =============================================================================

def normal_cdf(z: np.ndarray | float) -> np.ndarray | float:
    """Fast approximation of the standard normal CDF (used for tail probabilities)."""
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


# =============================================================================
# CSV ingestion (upload-only bytes)
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

def load_fp_from_bytes(content: bytes, pos_hint: Optional[str]) -> pd.DataFrame:
    """Parse a FantasyPros-style CSV (as bytes) into canonical columns."""
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

def load_fp_uploads(qb_bytes: Optional[bytes], flx_bytes: Optional[bytes],
                    k_bytes: Optional[bytes], dst_bytes: Optional[bytes]) -> pd.DataFrame:
    """Combine uploaded projections (as raw bytes) into one DataFrame."""
    frames: List[pd.DataFrame] = []
    if qb_bytes is not None:
        frames.append(load_fp_from_bytes(qb_bytes, pos_hint="QB"))
    if flx_bytes is not None:
        frames.append(load_fp_from_bytes(flx_bytes, pos_hint=None))
    if k_bytes is not None:
        frames.append(load_fp_from_bytes(k_bytes, pos_hint="K"))
    if dst_bytes is not None:
        frames.append(load_fp_from_bytes(dst_bytes, pos_hint="DST"))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def load_adp_upload(adp_bytes: Optional[bytes]) -> pd.DataFrame:
    """Parse an overall ADP CSV (as bytes) into canonical columns."""
    if adp_bytes is None:
        return pd.DataFrame()
    df = pd.read_csv(io.BytesIO(adp_bytes))
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
    """Expected games played given risk and position-specific episode parameters."""
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


def build_model_df(
    df: pd.DataFrame,
    league: LeagueConfig,
    var: VarianceModel,
    inj: InjuryModel,
    risk_by_pos: Dict[str, float],
    cv_by_pos: Dict[str, float],
    adp_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Create the core modeling DataFrame with EV above replacement and availability sigma."""
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
    if var.use_adp_sigma and "SigmaADP" in d.columns:
        var_sigma = pd.to_numeric(d["SigmaADP"], errors="coerce")
    else:
        var_sigma = None
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


# =============================================================================
# Draft math & recommendations
# =============================================================================

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

def compute_targets(
    model_df: pd.DataFrame,
    round_num: int,
    league: LeagueConfig,
    var: VarianceModel,
    top_n: int = 12,
    unavailable_pids: Optional[Set[str]] = None,
    risk_lambda: float = 0.15,
) -> pd.DataFrame:
    """Return top-N targets for the current pick (risk-adjusted, with next-pick lookahead)."""
    if model_df is None or model_df.empty:
        return pd.DataFrame()

    pick_now = pick_number(round_num, league.teams, league.slot)
    next_round = min(league.rounds, round_num + 1)
    pick_next = pick_number(next_round, league.teams, league.slot)

    df = model_df.copy()
    unavailable = set(unavailable_pids or set())
    if "PID" in df.columns and unavailable:
        df = df[~df["PID"].isin(unavailable)].copy()

    pool = expand_window_until(df, "ADP", pick_now, want=max(top_n*2, 30), base_window=12, max_window=80)
    pool = include_fallers(pool, df, pick_now, max_fallers=12)

    pool["ValueNow"] = pool["PreWeeklyEV_base"].astype(float)
    pool["PAvailNext"] = tail_prob(pick_next, pool["ADP"].values, pool["Sigma"].values).clip(0.0, 1.0)

    # For each position, estimate what's near your next pick as the "alternative"
    alt_vals: Dict[str, float] = {}
    for pos in ["QB", "RB", "WR", "TE", "K", "DST"]:
        near_next = expand_window_until(df[df["Pos"] == pos], "ADP", pick_next, want=max(10, top_n), base_window=12, max_window=80)
        alt_vals[pos] = float(near_next["PreWeeklyEV_base"].quantile(0.85)) if not near_next.empty else 0.0

    drop = np.maximum(0.0, pool["ValueNow"].values - pool["Pos"].map(alt_vals).values)
    regret = (1.0 - pool["PAvailNext"].values) * drop

    season_sd = pool["WeeklySD"].values * np.sqrt(np.maximum(pool["ProjG"].values, 1.0))
    lam = float(risk_lambda)
    pool["RiskAdj"] = pool["ValueNow"].values - lam * season_sd + 0.5 * regret

    pool["RoundNow"] = round_num
    pool["PickNow"] = pick_now
    pool["NextPick"] = pick_next

    pool = pool.sort_values(["RiskAdj", "ValueNow", "ADP"], ascending=[False, False, True]).reset_index(drop=True)
    return pool.head(top_n)
