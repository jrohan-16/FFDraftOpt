# -*- coding: utf-8 -*-
"""
Fantasy Draft Optimizer — Streamlit MVP v2.1 (critical fixes)

Focus of this update (no new "features", just fixes & hardening):
- **Removed all default file paths/storage**: the app no longer reads from or writes to /mnt/data;
  users must upload CSVs. (No temp saves of uploads.)
- **Per-game vs availability bug fixed**: PerGame is now derived from FPTS/G if present, else ProjPts/17,
  independent of expected games (ProjG). Season totals adjusted by risk are now AdjPts = PerGame * ProjG.
- **Beam planner pick-index bug fixed**: planning now deduplicates by Player name across rounds (not pool index).
- **Availability sigma fallback hardened**: if SigmaADP missing/NaN for some players, we fill with linear or constant σ.
- **CSV ingestion hardened**: flexible detection of 'FPTS' and 'FPTS/G'; tolerant column mapping for Player/Pos/Team/Bye.
- **No reliance on local files**: ADP is optional; if absent, ADP is proxied from projections order.

This file is a single-file Streamlit app.
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
# Helpers
# =============================================================================

def normal_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF without relying on np.erf or scipy.
    Uses the Abramowitz–Stegun 7.1.26 approximation for erf for numerical stability.
    """
    x = np.asarray(x, dtype=float)
    y = x / np.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * np.abs(y))
    # Horner's method for the polynomial
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t
    erf_y = 1.0 - poly * np.exp(-y * y)
    erf_y = np.where(y >= 0.0, erf_y, -erf_y)
    return 0.5 * (1.0 + erf_y)


def first_present(ls: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    """Return the first candidate that appears in 'ls' (case-insensitive)."""
    s = {c.lower() for c in ls}
    for c in candidates:
        if c.lower() in s:
            return c
    return None


def cols_lower(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase and strip column names for flexible matching."""
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]
    lower = {c: c.lower() for c in out.columns}
    return out.rename(columns=lower)


def pick_number(round_num: int, teams: int, slot: int) -> int:
    """
    Convert (round, slot) to overall pick number under snake draft.
    Round 1 is 1..teams; round 2 is teams..1; etc.
    """
    if round_num % 2 == 1:  # odd rounds
        return (round_num - 1) * teams + slot
    else:
        return round_num * teams - slot + 1


# =============================================================================
# CSV ingestion (uploads only)
# =============================================================================

def find_fpts_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """Return (season_total_col, per_game_col) heuristically."""
    cand = [c for c in df.columns if "fpts" in c.lower() or "fantasy points" in c.lower() or c.lower() == "pts"]
    if not cand:
        return None, None
    # Identify per-game variants
    per_game = [c for c in cand if "/g" in c.lower() or "per game" in c.lower() or c.lower().endswith("/g")]
    seasonish = [c for c in cand if c not in per_game]
    # Heuristics: prefer season totals that don't include "/g"
    season_col = sorted(seasonish, key=lambda c: len(c))[:1]
    per_game_col = sorted(per_game, key=lambda c: len(c))[:1]
    return (season_col[0] if season_col else None, per_game_col[0] if per_game_col else None)


def find_team_col(df: pd.DataFrame) -> Optional[str]:
    for cand in ["team", "tm", "teams", "nfl team", "nfl_team", "nflteam"]:
        if cand in df.columns:
            return cand
    return None


def _read_csv_from_upload(upload) -> pd.DataFrame:
    """Read a Streamlit UploadedFile into a DataFrame safely, without disk writes."""
    if upload is None:
        raise ValueError("No file provided")
    try:
        # Read bytes and pass a BytesIO buffer so multiple reads are safe
        data = upload.getvalue()
        return pd.read_csv(io.BytesIO(data))
    except Exception as e:
        raise ValueError(f"Failed to read CSV: {e}")


def load_fp_csv_upload(upload, pos_hint: Optional[str] = None) -> pd.DataFrame:
    """
    Load a FantasyPros projections CSV upload and normalize to:
    Player, Pos, Team, Bye, ProjPts, (optional) PerGame_hint, ADP/ECR/AVG if present.
    """
    raw = _read_csv_from_upload(upload)
    df = cols_lower(raw)

    # Player
    player_col = first_present(df.columns, ["player", "name", "player name"])
    if player_col is None:
        player_col = list(df.columns)[0]

    # Team & Bye
    team_col = find_team_col(df) or ""
    bye_col = first_present(df.columns, ["bye", "bye week", "bye week number"])

    # FPTS (season & per game)
    season_col, per_game_col = find_fpts_cols(df)
    if season_col is None and per_game_col is None:
        # Allow projections with only per-game or only season totals
        proj_season = pd.Series(0.0, index=df.index, dtype=float)
        per_game_hint = pd.Series(0.0, index=df.index, dtype=float)
    else:
        proj_season = pd.to_numeric(df[season_col], errors="coerce").fillna(0.0) if season_col else pd.Series(0.0, index=df.index, dtype=float)
        per_game_hint = pd.to_numeric(df[per_game_col], errors="coerce").fillna(0.0) if per_game_col else pd.Series(0.0, index=df.index, dtype=float)

    # Position
    if pos_hint is None:
        pos_col = first_present(df.columns, ["pos", "position"])
        if pos_col is None:
            raise ValueError("Could not infer position; please provide a file with a 'Pos' column or upload a position-specific file.")
        pos = df[pos_col].astype(str).str.upper().str.replace(" ", "", regex=False).replace({"DEF": "DST"})
    else:
        pos = pos_hint

    out = pd.DataFrame(
        {
            "Player": df[player_col].astype(str).str.strip(),
            "Pos": pos if isinstance(pos, str) else pos.astype(str),
            "Team": df[team_col].astype(str).str.upper().str.strip() if team_col in df.columns else "",
            "Bye": df[bye_col] if (bye_col is not None and bye_col in df.columns) else np.nan,
            "ProjPts": proj_season.astype(float),
            "PerGame_hint": per_game_hint.astype(float),
        }
    )
    # Optional ADP/ECR columns if present
    for adp_col in ["adp", "avg", "ecr"]:
        if adp_col in df.columns:
            out[adp_col.upper()] = pd.to_numeric(df[adp_col], errors="coerce")

    return out


def parse_adp_csv_upload(upload) -> pd.DataFrame:
    """
    Parse an ADP CSV upload into columns:
    Player, ADP, (optional) Pos, SigmaADP (from Std Dev or Min/Max range).
    """
    raw = _read_csv_from_upload(upload)
    df = cols_lower(raw)

    # Player
    player_col = first_present(df.columns, ["player", "name", "player name", "player team (pos)"])
    if player_col is None:
        player_col = list(df.columns)[0]

    # Position
    pos_col = first_present(df.columns, ["pos", "position"])
    pos_series = None
    if pos_col is not None:
        pos_series = df[pos_col].astype(str).str.upper().str.replace(" ", "", regex=False).replace({"DEF": "DST"})
    else:
        tmp = df[player_col].astype(str)
        if tmp.str.contains(r"\(", regex=True).any():
            pos_series = tmp.str.extract(r"\(([^)]+)\)")[0].str.upper().str.replace(" ", "", regex=False).replace({"DEF": "DST"})

    # ADP
    adp_col = first_present(df.columns, ["adp", "avg", "average", "overall"])
    if adp_col is None:
        raise ValueError("ADP file must have an ADP/AVG/Overall column.")

    # Sigma / range
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
    use_adp_sigma: bool = True            # prefer Sigma from ADP file if provided
    use_linear_sigma: bool = True         # else σ = a + b*ADP
    sigma_const: float = 12.0             # if not linear
    sigma_a: float = 3.0
    sigma_b: float = 0.04
    sigma_min: float = 6.0
    sigma_max: float = 22.0
    window: int = 12  # picks
    use_logistic: bool = False            # availability model choice
    logistic_scale_from_sigma: bool = True
    logistic_scale_const: float = 7.0     # if not from sigma


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
    """
    Convert 'Risk' into expected games using a simple hazard model.
    Expected available games ≈ 17 / (1 + h*L), where h scales with risk.
    """
    out = np.zeros_like(risk, dtype=float)
    for p in np.unique(pos):
        m = inj.miss_at_risk05.get(p, 2.0)  # missed games at risk=0.5
        L = inj.episode_len.get(p, 2.0)     # episode length (games)
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
    """
    Build the master model table with projections, ADP, risk, expected games,
    per-game scoring, replacement levels, VBD, PreDraft weekly EV baseline,
    and tiering by position.
    """
    d = df.copy()

    # Risk default if missing
    if "Risk" not in d.columns:
        d["Risk"] = d["Pos"].map(risk_by_pos).fillna(0.5)
    d["Risk"] = pd.to_numeric(d["Risk"], errors="coerce").fillna(0.5).clip(0.05, 0.95)

    # ADP: prefer explicit ADP, else join to uploaded ADP file, else proxy by ProjPts rank
    if adp_df is not None:
        if "Pos" in adp_df.columns and adp_df["Pos"].notna().any():
            cols = ["Player", "Pos", "ADP"] + (["SigmaADP"] if "SigmaADP" in adp_df.columns else [])
            d = d.merge(adp_df[cols], on=["Player", "Pos"], how="left", suffixes=("", "_adpfile"))
        else:
            cols = ["Player", "ADP"] + (["SigmaADP"] if "SigmaADP" in adp_df.columns else [])
            d = d.merge(adp_df[cols], on="Player", how="left", suffixes=("", "_adpfile"))

        if "ADP_adpfile" in d.columns:
            d["ADP"] = d["ADP"].combine_first(d["ADP_adpfile"])
            d.drop(columns=["ADP_adpfile"], inplace=True)

        if "SigmaADP_adpfile" in d.columns and "SigmaADP" not in d.columns:
            d["SigmaADP"] = d["SigmaADP_adpfile"]
            d.drop(columns=["SigmaADP_adpfile"], inplace=True)

    if "ADP" not in d.columns or d["ADP"].isna().all():
        d = d.sort_values(["ProjPts"], ascending=False).reset_index(drop=True)
        d["ADP"] = (np.arange(1, len(d) + 1)).astype(float)

    # Expected games via injury model
    pos_arr = d["Pos"].values.astype(str)
    d["ProjG"] = markov_proj_g(d["Risk"].values.astype(float), pos_arr, inj)

    # Per-game scoring (CRITICAL FIX): base per-game independent of ProjG
    # Prefer per-game hint from file, else ProjPts / 17
    if "PerGame_hint" in d.columns and d["PerGame_hint"].notna().any() and (d["PerGame_hint"].sum() > 0):
        d["PerGame"] = pd.to_numeric(d["PerGame_hint"], errors="coerce").fillna(0.0)
    else:
        d["PerGame"] = d["ProjPts"].astype(float) / 17.0

    # Weekly SD from per-game and CV by position
    d["WeeklySD"] = d["PerGame"] * d["Pos"].map(cv_by_pos).fillna(0.5)

    # Replacement levels (season & weekly)
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

    # Season replacement uses risk-adjusted totals
    d["AdjPts"] = d["PerGame"] * d["ProjG"]
    repl_pts: Dict[str, float] = {}
    for p in ["QB", "RB", "WR", "TE", "K", "DST"]:
        subset = d[d["Pos"] == p].copy()
        if subset.empty:
            repl_pts[p] = 0.0
            continue
        subset = subset.sort_values("AdjPts", ascending=False)
        # Use ceil to avoid being overly optimistic with fractional FLEX shares
        brank = int(np.ceil(league_starters.get(p, 0)))
        brank = max(1, min(len(subset), brank))
        repl_pts[p] = float(subset.iloc[brank - 1]["AdjPts"])

    # Streaming boost for K/DST (season points add ~17*boost)
    repl_pts["K"] = repl_pts.get("K", 0.0) + 17.0 * league.stream_boost_k
    repl_pts["DST"] = repl_pts.get("DST", 0.0) + 17.0 * league.stream_boost_dst

    d["ReplAdjPts"] = d["Pos"].map(repl_pts).fillna(0.0)
    d["VBD"] = (d["AdjPts"] - d["ReplAdjPts"]).clip(lower=0.0)

    # Weekly replacement from per-game
    repl_ppg: Dict[str, float] = {}
    for p in ["QB", "RB", "WR", "TE", "K", "DST"]:
        subset = d[d["Pos"] == p].copy()
        if subset.empty:
            repl_ppg[p] = 0.0
            continue
        subset = subset.sort_values("PerGame", ascending=False)
        brank = int(np.ceil(league_starters.get(p, 0)))
        brank = max(1, min(len(subset), brank))
        repl_ppg[p] = float(subset.iloc[brank - 1]["PerGame"])
    repl_ppg["K"] = repl_ppg.get("K", 0.0) + league.stream_boost_k
    repl_ppg["DST"] = repl_ppg.get("DST", 0.0) + league.stream_boost_dst
    d["ReplPPG"] = d["Pos"].map(repl_ppg).fillna(0.0)

    # Availability sigma (hardened fallback)
    sigma = None
    if var.use_adp_sigma and "SigmaADP" in d.columns:
        sigma = pd.to_numeric(d["SigmaADP"], errors="coerce")

    if sigma is None:
        sigma = pd.Series(np.nan, index=d.index, dtype=float)

    # fill NaNs with linear or constant
    if var.use_linear_sigma:
        sigma_fallback = var.sigma_a + var.sigma_b * d["ADP"].values
    else:
        sigma_fallback = np.full(len(d), var.sigma_const, dtype=float)
    sigma = sigma.fillna(pd.Series(sigma_fallback, index=d.index))
    d["Sigma"] = np.clip(sigma.values, var.sigma_min, var.sigma_max)

    if var.use_logistic:
        if var.logistic_scale_from_sigma:
            d["LogisticScale"] = d["Sigma"] * (np.sqrt(3.0) / np.pi)
        else:
            d["LogisticScale"] = var.logistic_scale_const

    # Pre-draft start share & EV base
    d["PreDraftStartShare"] = d["Pos"].map(start_share).fillna(1.0)
    d["PreWeeklyEV_base"] = np.maximum(0.0, d["PerGame"] - d["ReplPPG"]) * d["ProjG"] * d["PreDraftStartShare"]

    # PosRank & Tiers per position (simple 4-tier split)
    d["PosRank"] = d.groupby("Pos")["VBD"].rank(ascending=False, method="min")

    def _tierer(sr: pd.Series) -> pd.Series:
        n = len(sr)
        ranks = sr.rank(ascending=True, method="min")
        t1, t2, t3 = 0.15 * n, 0.35 * n, 0.65 * n
        tiers = 1 + (ranks > t1).astype(int) + (ranks > t2).astype(int) + (ranks > t3).astype(int)
        return tiers

    d["Tier"] = d.groupby("Pos")["PosRank"].transform(_tierer)

    return d


def tail_prob(pick: int, adp: np.ndarray, model_df: pd.DataFrame, var: VarianceModel) -> np.ndarray:
    """Compute P(available at 'pick') from ADP +/- noise under the chosen model."""
    if var.use_logistic and "LogisticScale" in model_df.columns:
        return 1.0 / (1.0 + np.exp((pick - adp) / model_df["LogisticScale"].values))
    else:
        return 1.0 - normal_cdf((pick - adp) / model_df["Sigma"].values)


def lineup_ppw_given(players: pd.DataFrame, league: LeagueConfig) -> float:
    """
    Compute lineup points-per-week (PPW) from a set of players with columns Pos, PerGame.
    Takes top starters at each position, then fills FLEX from remaining RB/WR/TE.
    """
    if players.empty:
        return 0.0
    pos = players["Pos"].values
    ppg = players["PerGame"].values

    def top_sum(mask: np.ndarray, n: int) -> Tuple[float, List[float]]:
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


def marginal_lineup_gain(
    model_df: pd.DataFrame, league: LeagueConfig, candidate_row: pd.Series, current_plus: Optional[pd.DataFrame] = None
) -> float:
    """
    Return the marginal increase in lineup PPW if we add 'candidate_row' to our current roster.
    Converted to season EV by multiplying by candidate's expected games (ProjG).
    """
    mine = model_df[model_df.get("Mine", 0) == 1][["Pos", "PerGame"]].copy()
    base_ppw = lineup_ppw_given(mine, league) if current_plus is None else lineup_ppw_given(current_plus, league)

    # add candidate to the tested roster
    if current_plus is None:
        mine_plus = pd.concat(
            [mine, pd.DataFrame([{"Pos": candidate_row["Pos"], "PerGame": candidate_row["PerGame"]}])], ignore_index=True
        )
    else:
        mine_plus = pd.concat(
            [current_plus, pd.DataFrame([{"Pos": candidate_row["Pos"], "PerGame": candidate_row["PerGame"]}])],
            ignore_index=True,
        )
    new_ppw = lineup_ppw_given(mine_plus, league)
    delta_ppw = max(0.0, new_ppw - base_ppw)
    return delta_ppw * float(candidate_row["ProjG"])


def compute_targets(
    model_df: pd.DataFrame, round_num: int, league: LeagueConfig, var: VarianceModel, use_marginal: bool = False
) -> pd.DataFrame:
    """Compute 'on the clock' targets for a given round with windowed candidate set around ADP."""
    pick = pick_number(round_num, league.teams, league.slot)
    df = model_df.copy()

    # Respect flags
    if "Taken" in df.columns:
        df = df[df["Taken"] == 0]

    # Window by ADP to focus on practical candidates
    df = df[np.abs(df["ADP"] - pick) <= var.window].copy()

    pav = tail_prob(pick, df["ADP"].values, df, var)
    df["PAvail"] = pav
    df["EV_Season"] = df["VBD"].values * pav

    if not use_marginal:
        df["WeeklyEV"] = df["PreWeeklyEV_base"].values * pav
    else:
        # roster-aware marginal delta
        df["WeeklyEV"] = [marginal_lineup_gain(model_df, league, row) * pav[i] for i, (_, row) in enumerate(df.iterrows())]

    cols = [
        "Player",
        "Pos",
        "Team",
        "PerGame",
        "VBD",
        "ADP",
        "ProjPts",
        "ProjG",
        "PAvail",
        "EV_Season",
        "ReplPPG",
        "WeeklyEV",
        "Tier",
        "PosRank",
    ]
    df = df.sort_values(["WeeklyEV", "Tier", "ADP"], ascending=[False, True, True])
    return df[[c for c in cols if c in df.columns]]


def weekly_sim(
    model_df: pd.DataFrame, league: LeagueConfig, num_sims: int = 500, use_corr: bool = False, rho_qb_wr: float = 0.2, rho_qb_te: float = 0.15
) -> pd.DataFrame:
    """
    Monte Carlo PPW for the current roster (Mine). If 'use_corr' True,
    add a crude team-level correlation between QB and his WR/TE.
    """
    df = model_df.copy()
    mine = df[df.get("Mine", 0) == 1].copy()
    if mine.empty:
        return pd.DataFrame({"PPW": []})

    mu = (mine["PerGame"].values * (mine["ProjG"].values / 17.0)).astype(float)
    sd = (mine["WeeklySD"].values * np.sqrt(mine["ProjG"].values / 17.0)).astype(float)
    pos = mine["Pos"].values
    team = mine["Team"].values

    sims: List[float] = []
    n = len(mine)

    if not use_corr:
        for _ in range(int(num_sims)):
            draws = np.random.normal(mu, sd, size=n)
            tmp = mine.copy()
            tmp["Draw"] = draws
            ppw = lineup_ppw_given(tmp[["Pos", "Draw"]].rename(columns={"Draw": "PerGame"}), league)
            sims.append(ppw)
        return pd.DataFrame({"PPW": sims})

    # Correlated version (team factor on QB)
    teams = sorted(set(team))
    team_idx = np.array([teams.index(t) for t in team])
    U = len(teams)
    rho = np.zeros(n)
    for i in range(n):
        if pos[i] == "QB":
            same_team = team[i]
            any_receiver = np.any((team == same_team) & ((pos == "WR") | (pos == "TE")))
            rho[i] = rho_qb_wr if any_receiver else 0.0
        else:
            rho[i] = 0.0

    for _ in range(int(num_sims)):
        z_team = np.random.normal(0, 1, size=U)
        eps = np.random.normal(0, 1, size=n)
        z = np.sqrt(1 - rho**2) * eps + rho * z_team[team_idx]
        draws = mu + sd * z
        tmp = mine.copy()
        tmp["Draw"] = draws
        ppw = lineup_ppw_given(tmp[["Pos", "Draw"]].rename(columns={"Draw": "PerGame"}), league)
        sims.append(ppw)
    return pd.DataFrame({"PPW": sims})


# -----------------------------
# Early-Round Planner (beam search with Player-name dedupe)
# -----------------------------

def planner_beam_search(
    model_df: pd.DataFrame,
    league: LeagueConfig,
    var: VarianceModel,
    rounds_plan: int = 8,
    pool_per_round: int = 12,
    beam_width: int = 50,
    exclude_k_dst_before: int = 10,
    max_qb_first_k: int = 1,
    use_marginal: bool = True,
) -> pd.DataFrame:
    """
    Greedy-ish beam search across the first 'rounds_plan' rounds.
    Deduplication is by Player name (not pool index) to prevent cross-round collisions.
    """
    # Prepare candidate pools per round
    pools: List[pd.DataFrame] = []
    for r in range(1, rounds_plan + 1):
        pick = pick_number(r, league.teams, league.slot)
        df = model_df[(model_df.get("Taken", 0) == 0)].copy()
        df = df[np.abs(df["ADP"] - pick) <= var.window]
        if r < exclude_k_dst_before:
            df = df[~df["Pos"].isin(["K", "DST"])]
        df = df.assign(PAvail=tail_prob(pick, df["ADP"].values, df, var))
        df = df.sort_values(["PreWeeklyEV_base", "PAvail"], ascending=[False, False])
        pools.append(df.head(pool_per_round).reset_index(drop=True))

    # Beam state: (cum_score, picks (as list of (round, player_name)), roster_df, qb_count, chosen_names_set)
    init_roster = model_df[model_df.get("Mine", 0) == 1][["Pos", "PerGame"]].copy()
    beam: List[Tuple[float, List[Tuple[int, str]], pd.DataFrame, int, set]] = [
        (0.0, [], init_roster, int((init_roster["Pos"] == "QB").sum()), set())
    ]

    for r in range(1, min(rounds_plan, len(pools)) + 1):
        next_states: List[Tuple[float, List[Tuple[int, str]], pd.DataFrame, int, set]] = []
        pool = pools[r - 1]

        for score, picks, roster_df, qb_cnt, chosen in beam:
            for i, row in pool.iterrows():
                pname = str(row["Player"])
                if pname in chosen:
                    continue
                pav = float(row["PAvail"])
                if r <= max_qb_first_k and row["Pos"] == "QB" and qb_cnt >= 1:
                    continue
                delta = marginal_lineup_gain(model_df, league, row, current_plus=roster_df)
                new_roster = pd.concat([roster_df, pd.DataFrame([{"Pos": row["Pos"], "PerGame": row["PerGame"]}])], ignore_index=True)
                new_qb_cnt = qb_cnt + (1 if row["Pos"] == "QB" else 0)
                new_score = score + pav * delta
                new_chosen = set(chosen); new_chosen.add(pname)
                next_states.append((new_score, picks + [(r, pname)], new_roster, new_qb_cnt, new_chosen))

        next_states.sort(key=lambda x: x[0], reverse=True)
        beam = next_states[:beam_width]
        if not beam:
            break

    # Render top sequences
    out_rows: List[Dict[str, object]] = []
    for rank, (score, picks, roster_df, qb_cnt, chosen) in enumerate(sorted(beam, key=lambda x: x[0], reverse=True), start=1):
        seq: List[str] = []
        total = 0.0
        roster_tmp = model_df[model_df.get("Mine", 0) == 1][["Pos", "PerGame"]].copy()
        for r, pname in picks:
            pool = pools[r - 1]
            row = pool[pool["Player"] == pname].iloc[0]
            pav = float(row["PAvail"])
            delta = marginal_lineup_gain(model_df, league, row, current_plus=roster_tmp)
            roster_tmp = pd.concat([roster_tmp, pd.DataFrame([{"Pos": row["Pos"], "PerGame": row["PerGame"]}])], ignore_index=True)
            total += pav * delta
            seq.append(f"R{r}: {row['Player']} ({row['Pos']}) | ADP {row['ADP']:.1f} | P(avail) {pav:.2f} | mΔEV {delta:.1f}")
        out_rows.append({"PlanRank": rank, "Total_ExpEV": round(total, 1), "Sequence": "  |  ".join(seq)})
    return pd.DataFrame(out_rows)


def snake_sim(model_df: pd.DataFrame, league: LeagueConfig, var: VarianceModel, rounds_sim: int = 8, runs: int = 200) -> pd.DataFrame:
    """
    Simulate snake drafts where the room picks by ADP + noise and you select
    best available by PreWeeklyEV_base. Returns per-run totals of your PreWeeklyEV.
    """
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
        sim_rank = np.argsort(np.argsort(draw))  # lower rank = earlier pick

        taken = np.zeros(N, dtype=bool)
        my_picks: List[int] = []

        for r in range(1, int(rounds_sim) + 1):
            pick = pick_number(r, league.teams, league.slot) - 1  # zero-based
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
    st.header("File uploads (required: at least one projections CSV)")
    up_qb = st.file_uploader("Projections — QB (CSV)", type=["csv"])
    up_flx = st.file_uploader("Projections — FLX (RB/WR/TE) (CSV)", type=["csv"])
    up_k = st.file_uploader("Projections — K (CSV)", type=["csv"])
    up_dst = st.file_uploader("Projections — DST/DEF (CSV)", type=["csv"])
    up_adp = st.file_uploader("ADP (CSV) — optional", type=["csv"])
    st.caption("Upload your FantasyPros CSVs. No local file fallbacks are used.")

# Prepare League/Variance/Injury configs
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
    use_adp_sigma=bool(use_adp_sigma),
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
)

inj = InjuryModel()

risk_by_pos = {"QB": float(r_qb), "RB": float(r_rb), "WR": float(r_wr), "TE": float(r_te), "K": float(r_k), "DST": float(r_dst)}
cv_by_pos = {"QB": float(cv_qb), "RB": float(cv_rb), "WR": float(cv_wr), "TE": float(cv_te), "K": float(cv_k), "DST": float(cv_dst)}
start_share = DEFAULT_START_SHARE if league.use_predraft_startshare else {p: 1.0 for p in DEFAULT_START_SHARE}

# Read uploads (no disk writes; no fallbacks)
frames: List[pd.DataFrame] = []
try:
    if up_qb is not None:
        frames.append(load_fp_csv_upload(up_qb, pos_hint="QB"))
    if up_flx is not None:
        frames.append(load_fp_csv_upload(up_flx, pos_hint=None))  # expects Pos in file
    if up_k is not None:
        frames.append(load_fp_csv_upload(up_k, pos_hint="K"))
    if up_dst is not None:
        frames.append(load_fp_csv_upload(up_dst, pos_hint="DST"))
except Exception as e:
    st.error(f"Failed to parse one of the uploaded projection files: {e}")
    st.stop()

if not frames:
    st.error("Please upload at least one projections CSV (QB / FLX / K / DST).")
    st.stop()

df_all = pd.concat(frames, ignore_index=True)
df_all = df_all.sort_values("ProjPts", ascending=False).drop_duplicates(["Player", "Pos"], keep="first").reset_index(drop=True)

# Promote ADP if included in projections
if "ADP" not in df_all.columns:
    if "AVG" in df_all.columns:
        df_all["ADP"] = df_all["AVG"]
    elif "ECR" in df_all.columns:
        df_all["ADP"] = df_all["ECR"]

# Optional ADP upload
adp_df = None
if up_adp is not None:
    try:
        adp_df = parse_adp_csv_upload(up_adp)
    except Exception as e:
        st.warning(f"Could not parse uploaded ADP: {e}")
        adp_df = None

# Build model table
model_df = build_model_df(df_all, league, var, inj, risk_by_pos, cv_by_pos, start_share, adp_df=adp_df)

# Apply flags from session
if "taken" not in st.session_state:
    st.session_state["taken"] = set()
if "mine" not in st.session_state:
    st.session_state["mine"] = set()
model_df["Taken"] = model_df["Player"].isin(st.session_state["taken"]).astype(int)
model_df["Mine"] = model_df["Player"].isin(st.session_state["mine"]).astype(int)

# Overview / sanity
st.subheader("Model overview")
st.caption("Snapshot of the model table (top 20 by ProjPts).")
st.dataframe(model_df.sort_values("ProjPts", ascending=False).head(20))

# -----------------------------
# On the clock
# -----------------------------
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
        st.success("Flags applied. Use any control to rerun and refresh targets.")
    if reset_taken:
        st.session_state["taken"] = set()
        st.success("Taken cleared.")
    if reset_mine:
        st.session_state["mine"] = set()
        st.success("Mine cleared.")

# -----------------------------
# Early-round planner
# -----------------------------
st.subheader("Early-round planner (beam search)")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    rounds_plan = st.number_input("Plan rounds", 2, min(12, int(rounds)), 6, 1)
with col2:
    pool_per_round = st.number_input("Pool per round", 5, 30, 12, 1)
with col3:
    beam_width = st.number_input("Beam width", 10, 200, 50, 5)
with col4:
    exclude_k_dst_before = st.number_input("Exclude K/DST before round", 1, 16, 10, 1)
with col5:
    max_qb_first_k = st.number_input("Max QBs in first K rounds", 0, 3, 1, 1)

use_marginal_planner = st.checkbox("Planner uses marginal WeeklyEV (recommended)", True)
if st.button("Run planner"):
    with st.spinner("Planning optimal sequences..."):
        plan_df = planner_beam_search(
            model_df,
            league,
            var,
            rounds_plan=int(rounds_plan),
            pool_per_round=int(pool_per_round),
            beam_width=int(beam_width),
            exclude_k_dst_before=int(exclude_k_dst_before),
            max_qb_first_k=int(max_qb_first_k),
            use_marginal=bool(use_marginal_planner),
        )
    st.dataframe(plan_df.head(20))
    st.download_button("Download plans (CSV)", plan_df.to_csv(index=False).encode("utf-8"), file_name="plans_beam.csv", mime="text/csv")

# -----------------------------
# Calibration
# -----------------------------
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

# -----------------------------
# SnakeSim
# -----------------------------
st.subheader("SnakeSim (others pick by ADP+noise; you by PreDraft WeeklyEV)")
colS1, colS2 = st.columns(2)
with colS1:
    rounds_sim = st.number_input("Simulate N rounds", 2, int(rounds), 8, 1)
with colS2:
    runs_sim = st.number_input("Runs", 50, 2000, 200, 50)
snake_df = snake_sim(model_df, league, var, rounds_sim=int(rounds_sim), runs=int(runs_sim))
st.write(f"Runs: {len(snake_df)} | Mean sum WeeklyEV over first {int(rounds_sim)} rounds: {snake_df['sum_wev'].mean():.1f}")
st.bar_chart(snake_df["sum_wev"])

# -----------------------------
# Exports
# -----------------------------
st.subheader("Exports")
csv = targets_df.to_csv(index=False).encode("utf-8")
st.download_button("Download targets (CSV)", csv, file_name="targets_round.csv", mime="text/csv")

state = {"taken": list(st.session_state["taken"]), "mine": list(st.session_state["mine"])}
st.download_button("Download flags (JSON)", json.dumps(state, indent=2).encode("utf-8"), file_name="draft_state.json", mime="application/json")

st.caption(
    "Uploads-only version (no local file fallbacks). Per-game & availability modeling corrected. "
    "Beam planner dedupes by Player across rounds and sigma fallback is robust."
)

