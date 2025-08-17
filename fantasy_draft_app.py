
import os, json, math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Helpers
# -----------------------------

def normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.erf(x / np.sqrt(2.0)))

def tail_prob_normal(pick: float, adp: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    # P(available) under Normal noise
    z = (pick - adp) / sigma
    return 1.0 - normal_cdf(z)

def tail_prob_logistic(pick: float, adp: np.ndarray, scale: np.ndarray) -> np.ndarray:
    # Logistic CDF F = 1/(1+exp(-(pick-ADP)/s)); P(available) = 1 - F
    z = (pick - adp) / scale
    return 1.0 / (1.0 + np.exp(z))

def first_present(ls: List[str], candidates: List[str]):
    s = set([c.lower() for c in ls])
    for c in candidates:
        if c.lower() in s:
            return c
    return None

def cols_lower(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    lower = {c: c.lower() for c in df.columns}
    return df.rename(columns=lower)

def pick_number(round_num: int, teams: int, slot: int) -> int:
    if round_num % 2 == 1:  # odd rounds
        return (round_num - 1) * teams + slot
    else:
        return round_num * teams - slot + 1

# -----------------------------
# FantasyPros ingestion
# -----------------------------

FP_FALLBACK_PATHS = [
    "/mnt/data/FantasyPros_Fantasy_Football_Projections_QB.csv",
    "/mnt/data/FantasyPros_Fantasy_Football_Projections_FLX.csv",
    "/mnt/data/FantasyPros_Fantasy_Football_Projections_K.csv",
    "/mnt/data/FantasyPros_Fantasy_Football_Projections_DST.csv",
]
ADP_FALLBACK_PATHS = [
    "/mnt/data/FantasyPros_2025_Overall_ADP_Rankings.csv"
]

def find_fpts_col(df: pd.DataFrame) -> Optional[str]:
    names = [c for c in df.columns if "fpts" in c.lower() or "fantasy points" in c.lower() or c.lower()=="pts"]
    if not names:
        return None
    names = sorted(names, key=lambda c: ("/g" in c.lower(), len(c)))  # prefer season total-ish
    return names[0]

def find_team_col(df: pd.DataFrame) -> Optional[str]:
    for cand in ["team", "tm", "teams", "NFL Team", "nfl team"]:
        if cand.lower() in df.columns:
            return cand.lower()
    return "team" if "team" in df.columns else None

def load_fp_csv(path: str, pos_hint: Optional[str]=None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = cols_lower(df)
    # columns
    player_col = first_present(list(df.columns), ["player", "name"])
    if player_col is None:
        raise ValueError(f"Could not find Player column in {path}")
    team_col = find_team_col(df) or "team"
    bye_col = first_present(list(df.columns), ["bye", "bye week", "bye_week"])
    fpts_col = find_fpts_col(df)
    if fpts_col is None:
        df["projpts"] = 0.0
    else:
        df["projpts"] = pd.to_numeric(df[fpts_col], errors="coerce").fillna(0.0)

    # position
    if pos_hint is None:
        pos_col = first_present(list(df.columns), ["pos", "position"])
        if pos_col is None:
            raise ValueError(f"Could not infer position for {path}. Provide pos_hint.")
        pos = df[pos_col].astype(str).str.upper().str.replace(" ", "").replace({"DEF":"DST"})
    else:
        pos = pos_hint

    out = pd.DataFrame({
        "Player": df[player_col].astype(str).str.strip(),
        "Pos": pos if isinstance(pos, str) else pos.astype(str),
        "Team": df[team_col].astype(str).str.upper().str.strip() if team_col in df.columns else "",
        "Bye": df[bye_col] if (bye_col is not None and bye_col in df.columns) else np.nan,
        "ProjPts": df["projpts"].astype(float),
    })
    # Optional ADP/ECR columns if present
    for adp_col in ["adp", "avg", "ecr"]:
        if adp_col in df.columns:
            out[adp_col.upper()] = pd.to_numeric(df[adp_col], errors="coerce")
    return out

def load_all_sources(user_files: List[Tuple[str, str]]) -> pd.DataFrame:
    frames = []

    for label, p in user_files:
        if p is None: continue
        hint = None
        if "qb" in label.lower():
            hint = "QB"
        elif "k" in label.lower():
            hint = "K"
        elif "dst" in label.lower() or "def" in label.lower():
            hint = "DST"
        df = load_fp_csv(p, pos_hint=hint)
        frames.append(df)

    if not frames:
        for p in FP_FALLBACK_PATHS:
            if os.path.exists(p):
                hint = None
                if "qb" in p.lower(): hint = "QB"
                elif "k.csv" in p.lower(): hint = "K"
                elif "dst" in p.lower(): hint = "DST"
                frames.append(load_fp_csv(p, pos_hint=hint))
        if not frames:
            st.stop()

    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all.sort_values("ProjPts", ascending=False).drop_duplicates(["Player","Pos"], keep="first")
    df_all = df_all.reset_index(drop=True)
    # Try to promote ADP if included
    if "ADP" not in df_all.columns:
        if "AVG" in df_all.columns:
            df_all["ADP"] = df_all["AVG"]
        elif "ECR" in df_all.columns:
            df_all["ADP"] = df_all["ECR"]
    return df_all

def parse_adp_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = cols_lower(df)
    # Try multiple schemas
    # Player
    player_col = first_present(list(df.columns), ["player", "name", "player name", "player team (pos)"])
    if player_col is None:
        # fallback: first column
        player_col = list(df.columns)[0]
    # Position
    pos_col = first_present(list(df.columns), ["pos", "position"])
    # If player includes "(POS)", try to extract POS
    pos_series = None
    if pos_col is not None:
        pos_series = df[pos_col].astype(str).str.upper().str.replace(" ", "").replace({"DEF":"DST"})
    else:
        tmp = df[player_col].astype(str)
        if tmp.str.contains(r"\(", regex=True).any():
            pos_series = tmp.str.extract(r"\(([^)]+)\)")[0].str.upper().str.replace(" ", "").replace({"DEF":"DST"})
    # ADP
    adp_col = first_present(list(df.columns), ["adp", "avg", "average", "overall"])
    if adp_col is None:
        raise ValueError("ADP file must have ADP/AVG/Overall column.")
    # Sigma
    sd_col = first_present(list(df.columns), ["std dev", "stdev", "stddev", "sd", "std"])
    min_col = first_present(list(df.columns), ["min pick", "best", "best pick", "min"])
    max_col = first_present(list(df.columns), ["max pick", "worst", "worst pick", "max"])

    out = pd.DataFrame({
        "Player": df[player_col].astype(str).str.strip(),
        "ADP": pd.to_numeric(df[adp_col], errors="coerce")
    })
    if pos_series is not None:
        out["Pos"] = pos_series

    sigma = None
    if sd_col is not None:
        sigma = pd.to_numeric(df[sd_col], errors="coerce")
    elif min_col is not None and max_col is not None:
        # Approximate sigma from range ~ 4*σ (95% interval) if range provided
        rng = pd.to_numeric(df[max_col], errors="coerce") - pd.to_numeric(df[min_col], errors="coerce")
        sigma = rng / 4.0
    if sigma is not None:
        out["SigmaADP"] = sigma

    return out

# -----------------------------
# Modeling
# -----------------------------

@dataclass
class LeagueConfig:
    teams: int = 12
    slot: int = 4
    rounds: int = 16
    starters: Dict[str,int] = None
    bench: int = 7
    flex_shares: Dict[str,float] = None
    use_predraft_startshare: bool = True
    stream_boost_k: float = 0.0
    stream_boost_dst: float = 0.0

    def __post_init__(self):
        if self.starters is None:
            self.starters = {"QB":1,"RB":2,"WR":2,"TE":1,"FLEX":1,"K":1,"DST":1}
        if self.flex_shares is None:
            self.flex_shares = {"RB":0.45,"WR":0.50,"TE":0.05}

@dataclass
class VarianceModel:
    use_linear_sigma: bool = True
    sigma_const: float = 12.0
    sigma_a: float = 3.0
    sigma_b: float = 0.04
    sigma_min: float = 6.0
    sigma_max: float = 22.0
    window: int = 12  # picks
    use_logistic: bool = False  # availability model
    logistic_scale_from_sigma: bool = True
    logistic_scale_const: float = 7.0  # if not from sigma
    use_adp_sigma: bool = True        # NEW: prefer Sigma from ADP file if provided

@dataclass
class InjuryModel:
    risk_alpha: float = 1.0
    miss_at_risk05: Dict[str,float] = None
    episode_len: Dict[str,float] = None

    def __post_init__(self):
        if self.miss_at_risk05 is None:
            self.miss_at_risk05 = {"QB":1.0,"RB":2.5,"WR":2.0,"TE":2.0,"K":0.8,"DST":0.2}
        if self.episode_len is None:
            self.episode_len = {"QB":1.5,"RB":2.5,"WR":2.0,"TE":2.0,"K":1.0,"DST":0.5}

DEFAULT_CV = {"QB":0.35,"RB":0.55,"WR":0.60,"TE":0.60,"K":0.30,"DST":0.25}
DEFAULT_START_SHARE = {"QB":0.95,"RB":0.65,"WR":0.60,"TE":0.50,"K":0.95,"DST":0.95}
DEFAULT_RISK_POS = {"QB":0.35,"RB":0.55,"WR":0.50,"TE":0.45,"K":0.20,"DST":0.10}

def markov_proj_g(risk: np.ndarray, pos: np.ndarray, inj: InjuryModel) -> np.ndarray:
    out = np.zeros_like(risk, dtype=float)
    for p in np.unique(pos):
        m = inj.miss_at_risk05.get(p, 2.0)
        L = inj.episode_len.get(p, 2.0)
        h05 = (m/17) / (1 - m/17) / L
        mask = (pos == p)
        rr = np.clip(risk[mask], 0.05, 0.95)
        h = h05 * np.power(rr/0.5, inj.risk_alpha)
        out[mask] = 17.0 / (1.0 + h*L)
    return out

def build_model_df(df: pd.DataFrame,
                   league: LeagueConfig,
                   var: VarianceModel,
                   inj: InjuryModel,
                   risk_by_pos: Dict[str,float],
                   cv_by_pos: Dict[str,float],
                   start_share: Dict[str,float],
                   adp_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    d = df.copy()

    # Risk default if missing
    if "Risk" not in d.columns:
        d["Risk"] = d["Pos"].map(risk_by_pos).fillna(0.5)
    d["Risk"] = pd.to_numeric(d["Risk"], errors="coerce").fillna(0.5).clip(0.05,0.95)

    # ADP: prefer explicit ADP, else join to uploaded ADP file, else proxy by ProjPts rank
    if adp_df is not None:
        # left join on Player+Pos if available
        if "Pos" in adp_df.columns and adp_df["Pos"].notna().any():
            d = d.merge(adp_df[["Player","Pos","ADP","SigmaADP"]].drop_duplicates("Player", keep="first") if "SigmaADP" in adp_df.columns else adp_df[["Player","Pos","ADP"]]
                        , on=["Player","Pos"], how="left", suffixes=("","_adpfile"))
        else:
            d = d.merge(adp_df[["Player","ADP","SigmaADP"]] if "SigmaADP" in adp_df.columns else adp_df[["Player","ADP"]]
                        , on="Player", how="left", suffixes=("","_adpfile"))
        d["ADP"] = d["ADP"].combine_first(d.get("ADP_adpfile"))
        if "SigmaADP" in d.columns:
            d["SigmaADP"] = d["SigmaADP"]
        elif "SigmaADP_adpfile" in d.columns:
            d["SigmaADP"] = d["SigmaADP_adpfile"]
        for col in ["ADP_adpfile","SigmaADP_adpfile"]:
            if col in d.columns: d = d.drop(columns=[col])

    if "ADP" not in d.columns or d["ADP"].isna().all():
        d = d.sort_values(["ProjPts"], ascending=False).reset_index(drop=True)
        d["ADP"] = (np.arange(1, len(d)+1)).astype(float)

    # ProjG via Markov
    pos_arr = d["Pos"].values.astype(str)
    d["ProjG"] = markov_proj_g(d["Risk"].values.astype(float), pos_arr, inj)

    # PerGame and WeeklySD
    d["PerGame"] = d["ProjPts"] / d["ProjG"].replace(0, np.nan)
    d["PerGame"] = d["PerGame"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    d["WeeklySD"] = d.apply(lambda r: r["PerGame"] * cv_by_pos.get(r["Pos"], 0.5), axis=1)

    # Replacement levels
    starters = league.starters.copy()
    flex_rb = league.teams * starters.get("FLEX",0) * league.flex_shares.get("RB",0.45)
    flex_wr = league.teams * starters.get("FLEX",0) * league.flex_shares.get("WR",0.50)
    flex_te = league.teams * starters.get("FLEX",0) * league.flex_shares.get("TE",0.05)
    league_starters = {
        "QB": league.teams*starters.get("QB",0),
        "RB": league.teams*starters.get("RB",0) + flex_rb,
        "WR": league.teams*starters.get("WR",0) + flex_wr,
        "TE": league.teams*starters.get("TE",0) + flex_te,
        "K":  league.teams*starters.get("K",0),
        "DST": league.teams*starters.get("DST",0),
    }
    repl_pts = {}
    for p in ["QB","RB","WR","TE","K","DST"]:
        subset = d[d["Pos"]==p].copy()
        if subset.empty:
            repl_pts[p] = 0.0
            continue
        subset["AdjPts"] = subset["ProjPts"] * (subset["ProjG"]/17.0)
        subset = subset.sort_values("AdjPts", descending:=False)
        subset = subset.sort_values("AdjPts", ascending=False)
        brank = int(round(league_starters.get(p,0)))
        brank = max(1, min(len(subset), brank))
        repl_pts[p] = float(subset.iloc[brank-1]["AdjPts"])
    # Streaming boost for K/DST
    repl_pts["K"]  = repl_pts.get("K",0.0)  + 17.0 * league.stream_boost_k
    repl_pts["DST"] = repl_pts.get("DST",0.0) + 17.0 * league.stream_boost_dst

    d["AdjPts"] = d["ProjPts"] * (d["ProjG"]/17.0)
    d["ReplPPG"] = d["Pos"].map({p: v/17.0 for p,v in repl_pts.items()}).fillna(0.0)
    d["VBD"] = d["AdjPts"] - d["Pos"].map(repl_pts).fillna(0.0)

    # Availability sigma and logistic scale
    if var.use_adp_sigma and "SigmaADP" in d.columns and d["SigmaADP"].notna().any():
        sigma = d["SigmaADP"].fillna(method="ffill").fillna(method="bfill").fillna(12.0).values
        sigma = np.clip(sigma, var.sigma_min, var.sigma_max)
    elif var.use_linear_sigma:
        sigma = np.clip(var.sigma_a + var.sigma_b * d["ADP"].values, var.sigma_min, var.sigma_max)
    else:
        sigma = np.full(len(d), var.sigma_const, dtype=float)
    d["Sigma"] = sigma
    if var.use_logistic:
        s = sigma * (np.sqrt(3.0) / math.pi) if var.logistic_scale_from_sigma else np.full(len(d), var.logistic_scale_const, float)
        d["LogisticScale"] = s

    # PreDraft start share & EV base
    d["PreDraftStartShare"] = d["Pos"].map(DEFAULT_START_SHARE).fillna(1.0)
    d["PreWeeklyEV_base"] = np.maximum(0.0, d["PerGame"] - d["ReplPPG"]) * d["ProjG"] * d["PreDraftStartShare"]

    # PosRank and Tier
    d["PosRank"] = d.groupby("Pos")["VBD"].rank(ascending=False, method="min")
    def tierer(sr):
        n = len(sr)
        ranks = sr.rank(ascending=True, method="min")
        t1, t2, t3 = 0.15*n, 0.35*n, 0.65*n
        tiers = 1 + (ranks>t1).astype(int) + (ranks>t2).astype(int) + (ranks>t3).astype(int)
        return tiers
    d["Tier"] = d.groupby("Pos")["PosRank"].transform(tierer)

    return d

def tail_prob(pick, adp, model_df: pd.DataFrame, var) -> np.ndarray:
    if var.use_logistic and "LogisticScale" in model_df.columns:
        return 1.0 / (1.0 + np.exp((pick - adp) / model_df["LogisticScale"].values))
    else:
        return 1.0 - normal_cdf((pick - adp) / model_df["Sigma"].values)

def lineup_ppw_given(players: pd.DataFrame, league) -> float:
    if players.empty:
        return 0.0
    pos = players["Pos"].values
    ppg = players["PerGame"].values
    def top_sum(mask, n):
        vals = sorted(ppg[mask], reverse=True)
        return sum(vals[:max(0,n)]), vals[max(0,n):]
    qb_s, _ = top_sum(pos=="QB", league.starters.get("QB",0))
    rb_s, rb_r = top_sum(pos=="RB", league.starters.get("RB",0))
    wr_s, wr_r = top_sum(pos=="WR", league.starters.get("WR",0))
    te_s, te_r = top_sum(pos=="TE", league.starters.get("TE",0))
    k_s, _ = top_sum(pos=="K", league.starters.get("K",0))
    d_s, _ = top_sum(pos=="DST", league.starters.get("DST",0))
    flex_pool = sorted(rb_r + wr_r + te_r, reverse=True)
    fx_s = sum(flex_pool[:league.starters.get("FLEX",0)])
    return qb_s + rb_s + wr_s + te_s + k_s + d_s + fx_s

def marginal_lineup_gain(model_df: pd.DataFrame, league, candidate_row: pd.Series, current_plus: Optional[pd.DataFrame]=None) -> float:
    # baseline lineup PPW from current Mine or provided current_plus
    mine = model_df[model_df.get("Mine",0)==1][["Pos","PerGame"]].copy()
    base_ppw = lineup_ppw_given(mine, league) if current_plus is None else lineup_ppw_given(current_plus, league)
    # add candidate
    if current_plus is None:
        mine_plus = pd.concat([mine, pd.DataFrame([{"Pos":candidate_row["Pos"], "PerGame":candidate_row["PerGame"]}])], ignore_index=True)
    else:
        mine_plus = pd.concat([current_plus, pd.DataFrame([{"Pos":candidate_row["Pos"], "PerGame":candidate_row["PerGame"]}])], ignore_index=True)
    new_ppw = lineup_ppw_given(mine_plus, league)
    delta_ppw = max(0.0, new_ppw - base_ppw)
    return delta_ppw * float(candidate_row["ProjG"])

def compute_targets(model_df: pd.DataFrame, round_num: int, league, var, use_marginal: bool=False) -> pd.DataFrame:
    pick = pick_number(round_num, league.teams, league.slot)
    df = model_df.copy()
    if "Taken" in df.columns:
        df = df[df["Taken"]==0]
    df = df[np.abs(df["ADP"] - pick) <= var.window].copy()

    pav = tail_prob(pick, df["ADP"].values, df, var)
    df["PAvail"] = pav
    df["EV_Season"] = df["VBD"].values * pav
    if not use_marginal:
        df["WeeklyEV"] = df["PreWeeklyEV_base"].values * pav
    else:
        df["WeeklyEV"] = [marginal_lineup_gain(model_df, league, row) * pav[i] for i, (_, row) in enumerate(df.iterrows())]

    cols = ["Player","Pos","Team","PerGame","VBD","ADP","ProjG","PAvail","EV_Season","ReplPPG","WeeklyEV","Tier","PosRank"]
    df = df.sort_values(["WeeklyEV","Tier","ADP"], ascending=[False, True, True])
    return df[cols]

def roster_summary(model_df: pd.DataFrame, league) -> Dict[str,float]:
    df = model_df.copy()
    if "Mine" not in df.columns or df["Mine"].sum()==0:
        return {"PPW":0.0,"PPW_repl":0.0,"WeeklyVBD":0.0,"SeasonPts":0.0,"SeasonVBD":0.0}
    mine = df[df["Mine"]==1].copy()
    ppw = lineup_ppw_given(mine[["Pos","PerGame"]], league)

    repl_ppg = df.groupby("Pos")["ReplPPG"].max().to_dict()
    ppw_repl = (
        repl_ppg.get("QB",0.0)*league.starters.get("QB",0) +
        repl_ppg.get("RB",0.0)*league.starters.get("RB",0) +
        repl_ppg.get("WR",0.0)*league.starters.get("WR",0) +
        repl_ppg.get("TE",0.0)*league.starters.get("TE",0) +
        repl_ppg.get("K",0.0)*league.starters.get("K",0) +
        repl_ppg.get("DST",0.0)*league.starters.get("DST",0) +
        league.starters.get("FLEX",0) * (
            repl_ppg.get("RB",0.0)*0.45 + repl_ppg.get("WR",0.0)*0.50 + repl_ppg.get("TE",0.0)*0.05
        )
    )
    weekly_vbd = ppw - ppw_repl
    return {"PPW":ppw, "PPW_repl":ppw_repl, "WeeklyVBD":weekly_vbd, "SeasonPts":ppw*17, "SeasonVBD":weekly_vbd*17}

def weekly_sim(model_df: pd.DataFrame, league, num_sims=500, use_corr=False, rho_qb_wr=0.2, rho_qb_te=0.15) -> pd.DataFrame:
    df = model_df.copy()
    mine = df[df.get("Mine",0)==1].copy()
    if mine.empty:
        return pd.DataFrame({"PPW":[]})
    mu = (mine["PerGame"].values * (mine["ProjG"].values/17.0)).astype(float)
    sd = (mine["WeeklySD"].values * np.sqrt(mine["ProjG"].values/17.0)).astype(float)
    pos = mine["Pos"].values
    team = mine["Team"].values

    sims = []
    n = len(mine)
    if not use_corr:
        for _ in range(int(num_sims)):
            draws = np.random.normal(mu, sd, size=n)
            tmp = mine.copy()
            tmp["Draw"] = draws
            ppw = lineup_ppw_given(tmp[["Pos","Draw"]].rename(columns={"Draw":"PerGame"}), league)
            sims.append(ppw)
        return pd.DataFrame({"PPW": sims})

    teams_unique, team_idx = np.unique(team, return_inverse=True)
    U = len(teams_unique)
    has_qb_team = {t: (("QB" in pos[team==t]) if np.any(team==t) else False) for t in teams_unique}
    rho = np.zeros(n, dtype=float)
    for i in range(n):
        if pos[i] == "WR":
            rho[i] = rho_qb_wr if has_qb_team[teams_unique[team_idx[i]]] else 0.0
        elif pos[i] == "TE":
            rho[i] = rho_qb_te if has_qb_team[teams_unique[team_idx[i]]] else 0.0
        elif pos[i] == "QB":
            same_team = teams_unique[team_idx[i]]
            any_receiver = np.any((team==same_team) & ((pos=="WR") | (pos=="TE")))
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
        ppw = lineup_ppw_given(tmp[["Pos","Draw"]].rename(columns={"Draw":"PerGame"}), league)
        sims.append(ppw)
    return pd.DataFrame({"PPW": sims})

# -----------------------------
# Early-Round Planner (beam search)
# -----------------------------

def planner_beam_search(model_df: pd.DataFrame, league, var, rounds_plan=8, pool_per_round=12, beam_width=50,
                        exclude_k_dst_before=10, max_qb_first_k=1, use_marginal=True):
    # Prepare candidate pools per round
    pools = []
    for r in range(1, rounds_plan+1):
        pick = pick_number(r, league.teams, league.slot)
        df = model_df[(model_df.get("Taken",0)==0)].copy()
        df = df[np.abs(df["ADP"] - pick) <= var.window]
        if r < exclude_k_dst_before:
            df = df[~df["Pos"].isin(["K","DST"])]
        # Rank by pre-WeeklyEV base (fast) and keep top pool_per_round
        df = df.assign(PAvail=tail_prob(pick, df["ADP"].values, df, var))
        if use_marginal:
            # Start from current roster only; marginal will be recomputed in the loop with growing roster
            df = df.sort_values(["PreWeeklyEV_base","PAvail"], ascending=[False, False])
        else:
            df = df.sort_values(["PreWeeklyEV_base","PAvail"], ascending=[False, False])
        pools.append(df.head(pool_per_round).reset_index(drop=True))

    # Beam state: (cum_score, picks_indices, roster_df (Pos, PerGame), qb_count)
    init_roster = model_df[model_df.get("Mine",0)==1][["Pos","PerGame"]].copy()
    beam = [ (0.0, [], init_roster.copy(), int((init_roster["Pos"]=="QB").sum())) ]

    for r in range(1, rounds_plan+1):
        pick = pick_number(r, league.teams, league.slot)
        pool = pools[r-1]
        new_beam = []
        for score, picks, roster_df, qb_cnt in beam:
            used_players = set(picks)
            for idx, row in pool.iterrows():
                # enforce no duplicate player and QB cap
                if idx in used_players:  # index local to pool; need a more robust identity
                    pass
                # Use global identity by player name + pos
                key = (row["Player"], row["Pos"])
                if key in [(pool.iloc[i]["Player"], pool.iloc[i]["Pos"]) for i in picks]:
                    continue
                if row["Pos"]=="QB" and qb_cnt >= max_qb_first_k:
                    continue
                # compute marginal
                if use_marginal:
                    delta = marginal_lineup_gain(model_df, league, row, current_plus=roster_df)
                else:
                    delta = max(0.0, row["PerGame"] - row["ReplPPG"]) * float(row["ProjG"])
                pav = float(row["PAvail"])
                new_score = score + pav * delta
                # update roster
                new_roster = pd.concat([roster_df, pd.DataFrame([{"Pos":row["Pos"], "PerGame":row["PerGame"]}])], ignore_index=True)
                new_qb_cnt = qb_cnt + (1 if row["Pos"]=="QB" else 0)
                new_picks = picks + [idx]
                new_beam.append((new_score, new_picks, new_roster, new_qb_cnt))
        # prune
        new_beam.sort(key=lambda x: x[0], reverse=True)
        beam = new_beam[:beam_width]
        if not beam:
            break

    # Build output
    out_rows = []
    for rank, (score, picks, roster_df, qb_cnt) in enumerate(sorted(beam, key=lambda x: x[0], reverse=True), start=1):
        seq = []
        total = 0.0
        roster_tmp = model_df[model_df.get("Mine",0)==1][["Pos","PerGame"]].copy()
        for r in range(1, min(rounds_plan, len(pools))+1):
            if r-1 >= len(picks): break
            pool = pools[r-1]
            row = pool.iloc[picks[r-1]]
            pav = float(row["PAvail"])
            delta = marginal_lineup_gain(model_df, league, row, current_plus=roster_tmp)
            roster_tmp = pd.concat([roster_tmp, pd.DataFrame([{"Pos":row["Pos"], "PerGame":row["PerGame"]}])], ignore_index=True)
            total += pav * delta
            seq.append(f"R{r}: {row['Player']} ({row['Pos']}) | ADP {row['ADP']:.1f} | P(avail) {pav:.2f} | mΔEV {delta:.1f}")
        out_rows.append({"PlanRank": rank, "Total_ExpEV": round(total,1), "Sequence": "  |  ".join(seq)})
    return pd.DataFrame(out_rows)

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Fantasy Draft Optimizer (Streamlit MVP v1.2)", layout="wide")
st.title("Fantasy Draft Optimizer — Streamlit MVP v1.2")

with st.sidebar:
    st.header("League Settings")
    teams = st.number_input("Teams", 8, 20, 12, 1)
    slot = st.number_input("Your Draft Slot", 1, teams, 4, 1)
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

    st.header("Availability model")
    use_lin = st.checkbox("Use linear sigma (a+b·ADP)", True)
    sigma_const = st.number_input("Constant sigma (if off)", 1.0, 40.0, 12.0, 0.5)
    sigma_a = st.number_input("Sigma a", 0.0, 30.0, 3.0, 0.1)
    sigma_b = st.number_input("Sigma b", 0.0, 0.2, 0.04, 0.005)
    sigma_min = st.number_input("Sigma min", 1.0, 60.0, 6.0, 1.0)
    sigma_max = st.number_input("Sigma max", 1.0, 80.0, 22.0, 1.0)
    window = st.number_input("ADP window (picks)", 0, 60, 12, 1)

    use_logistic = st.checkbox("Use logistic P(Avail) (vs Normal)", False)
    logistic_from_sigma = st.checkbox("Logistic scale from sigma (s = σ·√3/π)", True)
    logistic_scale_const = st.number_input("Logistic scale (if constant)", 1.0, 40.0, 7.0, 0.5)
    use_adp_sigma = st.checkbox("Use per-player σ from ADP file (if present)", True)

    st.header("Injury & variance")
    risk_alpha = st.number_input("Risk→Hazard alpha", 0.1, 3.0, 1.0, 0.1)
    st.caption("Higher alpha increases penalty for high-risk players.")
    st.markdown("**Risk by position (default if per-player missing)**")
    r_qb = st.slider("Risk QB", 0.0, 1.0, 0.35, 0.05)
    r_rb = st.slider("Risk RB", 0.0, 1.0, 0.55, 0.05)
    r_wr = st.slider("Risk WR", 0.0, 1.0, 0.50, 0.05)
    r_te = st.slider("Risk TE", 0.0, 1.0, 0.45, 0.05)
    r_k = st.slider("Risk K", 0.0, 1.0, 0.20, 0.05)
    r_dst = st.slider("Risk DST", 0.0, 1.0, 0.10, 0.05)

    st.markdown("**Weekly volatility (CV by position)**")
    cv_qb = st.slider("CV QB", 0.0, 1.0, 0.35, 0.05)
    cv_rb = st.slider("CV RB", 0.0, 1.0, 0.55, 0.05)
    cv_wr = st.slider("CV WR", 0.0, 1.0, 0.60, 0.05)
    cv_te = st.slider("CV TE", 0.0, 1.0, 0.60, 0.05)
    cv_k = st.slider("CV K", 0.0, 1.0, 0.30, 0.05)
    cv_dst = st.slider("CV DST", 0.0, 1.0, 0.25, 0.05)

    st.header("Monte Carlo")
    num_sims = st.number_input("WeeklySim runs", 100, 10000, 800, 100)
    use_corr = st.checkbox("Use team-shock correlation", True)
    rho_qb_wr = st.slider("Corr QB–WR", 0.0, 0.6, 0.20, 0.05)
    rho_qb_te = st.slider("Corr QB–TE", 0.0, 0.6, 0.15, 0.05)

    st.header("SnakeSim")
    rounds_sim = st.number_input("Early rounds to simulate", 2, 15, 8, 1)
    runs_sim = st.number_input("SnakeSim runs", 50, 5000, 400, 50)

# -----------------------------
# Data upload
# -----------------------------

st.subheader("Load projections (FantasyPros CSVs)")
col1, col2, col3, col4 = st.columns(4)
f_qb = col1.file_uploader("QB CSV")
f_flx = col2.file_uploader("FLX (RB/WR/TE) CSV")
f_k = col3.file_uploader("K CSV")
f_dst = col4.file_uploader("DST CSV")

st.subheader("ADP/ECR CSV")
f_adp = st.file_uploader("ADP/ECR CSV (Player, [Pos], ADP ± Std Dev)", key="adp_file")
if f_adp is None:
    # try fallback path (user's uploaded file saved server-side)
    for p in ADP_FALLBACK_PATHS:
        if os.path.exists(p):
            st.caption(f"Using ADP fallback: {p}")
            f_adp = open(p, "rb")
            break

user_files = []
if f_qb is not None:
    tmp_qb = f"/tmp/qb_{getattr(f_qb, 'name', 'qb.csv')}"; open(tmp_qb, "wb").write(f_qb.getbuffer()); user_files.append(("QB", tmp_qb))
if f_flx is not None:
    tmp_flx = f"/tmp/flx_{getattr(f_flx, 'name', 'flx.csv')}"; open(tmp_flx, "wb").write(f_flx.getbuffer()); user_files.append(("FLX", tmp_flx))
if f_k is not None:
    tmp_k = f"/tmp/k_{getattr(f_k, 'name', 'k.csv')}"; open(tmp_k, "wb").write(f_k.getbuffer()); user_files.append(("K", tmp_k))
if f_dst is not None:
    tmp_dst = f"/tmp/dst_{getattr(f_dst, 'name', 'dst.csv')}"; open(tmp_dst, "wb").write(f_dst.getbuffer()); user_files.append(("DST", tmp_dst))

try:
    df_raw = load_all_sources(user_files)
except Exception as e:
    st.error(f"Failed to load projections: {e}")
    st.stop()

adp_df = None
if f_adp is not None:
    try:
        if hasattr(f_adp, "read"):
            buf = f_adp.read()
            tmp_adp = "/tmp/adp_upload.csv"; open(tmp_adp, "wb").write(buf)
        else:
            # already a path-like from fallback
            tmp_adp = f_adp.name
        adp_df = parse_adp_csv(tmp_adp)
        st.success(f"ADP loaded: {len(adp_df)} rows. Sigma present: {'SigmaADP' in adp_df.columns and adp_df['SigmaADP'].notna().any()}")
    except Exception as e:
        st.warning(f"Could not parse ADP/ECR file: {e}"); adp_df = None

st.write(f"Loaded {len(df_raw)} player rows.")
st.dataframe(df_raw.head(10))

# -----------------------------
# Build model table
# -----------------------------

league = LeagueConfig(
    teams=int(teams), slot=int(slot), rounds=int(rounds),
    starters={"QB":int(s_qb),"RB":int(s_rb),"WR":int(s_wr),"TE":int(s_te),"FLEX":int(s_flex),"K":int(s_k),"DST":int(s_dst)},
    bench=int(bench),
    flex_shares={"RB":float(flex_rb),"WR":float(flex_wr),"TE":float(flex_te)},
    use_predraft_startshare=True if use_predraft_ss else False,
    stream_boost_k=float(stream_k),
    stream_boost_dst=float(stream_dst)
)

var = VarianceModel(
    use_linear_sigma=bool(use_lin),
    sigma_const=float(sigma_const),
    sigma_a=float(sigma_a),
    sigma_b=float(sigma_b),
    sigma_min=float(sigma_min),
    sigma_max=float(sigma_max),
    window=int(window),
    use_logistic=bool(use_logistic),
    logistic_scale_from_sigma=bool(logistic_from_sigma),
    logistic_scale_const=float(logistic_scale_const),
    use_adp_sigma=bool(use_adp_sigma)
)

inj = InjuryModel(risk_alpha=float(risk_alpha))

risk_by_pos = {"QB":float(r_qb),"RB":float(r_rb),"WR":float(r_wr),"TE":float(r_te),"K":float(r_k),"DST":float(r_dst)}
cv_by_pos = {"QB":float(cv_qb),"RB":float(cv_rb),"WR":float(cv_wr),"TE":float(cv_te),"K":float(cv_k),"DST":float(cv_dst)}
start_share = DEFAULT_START_SHARE if use_predraft_ss else {k:1.0 for k in DEFAULT_START_SHARE}

model_df = build_model_df(df_raw, league, var, inj, risk_by_pos, cv_by_pos, start_share, adp_df=adp_df)

# Persist flags
if "taken" not in st.session_state:
    st.session_state["taken"] = set()
if "mine" not in st.session_state:
    st.session_state["mine"] = set()

model_df["Taken"] = model_df["Player"].isin(st.session_state["taken"]).astype(int)
model_df["Mine"] = model_df["Player"].isin(st.session_state["mine"]).astype(int)

# -----------------------------
# On the clock
# -----------------------------
st.subheader("On the clock")
round_num = st.number_input("Round", 1, int(rounds), 1, 1)
use_marginal = st.checkbox("Rank by roster-based marginal WeeklyEV (delta lineup PPW)", True)
targets = compute_targets(model_df, int(round_num), league, var, use_marginal=use_marginal)
st.dataframe(targets.head(60))

with st.expander("Mark picks (Taken/Mine)"):
    taken_add = st.multiselect("Mark TAKEN", list(targets["Player"].values), key="taken_ms")
    mine_add = st.multiselect("Mark MINE", list(targets["Player"].values), key="mine_ms")
    apply = st.button("Apply flags", key="apply_flags")
    if apply:
        st.session_state["taken"].update(taken_add)
        st.session_state["mine"].update(mine_add)
        st.success("Flags applied. Rerun to update lists.")

# -----------------------------
# Roster summary & WeeklySim
# -----------------------------
st.subheader("Roster summary & WeeklySim")
summary = roster_summary(model_df, league)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Lineup PPW", f"{summary['PPW']:.2f}")
c2.metric("Repl PPW", f"{summary['PPW_repl']:.2f}")
c3.metric("Weekly VBD", f"{summary['WeeklyVBD']:.2f}")
c4.metric("Season Pts", f"{summary['SeasonPts']:.0f}")
c5.metric("Season VBD", f"{summary['SeasonVBD']:.0f}")

if summary["PPW"]>0:
    sims_df = weekly_sim(model_df, league, num_sims=int(num_sims), use_corr=bool(use_corr), rho_qb_wr=float(rho_qb_wr), rho_qb_te=float(rho_qb_te))
    st.write(f"WeeklySim runs: {len(sims_df)}")
    st.line_chart(sims_df["PPW"])
    q10, q25, q50, q75, q90 = np.percentile(sims_df["PPW"], [10,25,50,75,90])
    st.write(f"PPW percentiles — P10: {q10:.2f}, P25: {q25:.2f}, P50: {q50:.2f}, P75: {q75:.2f}, P90: {q90:.2f}")

# -----------------------------
# Early-Round Planner
# -----------------------------
st.subheader("Early-Round Planner (Beam Search)")
colp1, colp2, colp3 = st.columns(3)
rounds_plan = colp1.number_input("Rounds to plan", 2, 12, 8, 1)
pool_per_round = colp2.number_input("Candidates per round", 5, 25, 12, 1)
beam_width = colp3.number_input("Beam width", 10, 200, 60, 10)
colp4, colp5 = st.columns(2)
exclude_k_dst_before = colp4.number_input("Exclude K/DST before round", 1, 16, 10, 1)
max_qb_first_k = colp5.number_input("Max QB in first K rounds", 0, 3, 1, 1)
use_marginal_planner = st.checkbox("Planner uses marginal WeeklyEV (recommended)", True)

if st.button("Run planner"):
    plan_df = planner_beam_search(model_df, league, var, rounds_plan=int(rounds_plan),
                                  pool_per_round=int(pool_per_round),
                                  beam_width=int(beam_width),
                                  exclude_k_dst_before=int(exclude_k_dst_before),
                                  max_qb_first_k=int(max_qb_first_k),
                                  use_marginal=bool(use_marginal_planner))
    st.dataframe(plan_df.head(10))
    # Export
    st.download_button("Download plans (CSV)", plan_df.to_csv(index=False).encode("utf-8"),
                       file_name="early_round_plans.csv", mime="text/csv")

# -----------------------------
# Calibration: P(Avail) curve for a selected player
# -----------------------------
st.subheader("Calibration: P(Avail) curve")
sel_player = st.selectbox("Player", options=model_df["Player"].unique().tolist()[:500])
sel_row = model_df[model_df["Player"]==sel_player].head(1)
if not sel_row.empty:
    adp_val = float(sel_row["ADP"].values[0])
    if var.use_logistic and "LogisticScale" in model_df.columns:
        scale_val = float(sel_row["LogisticScale"].values[0])
        curve = [ 1.0 / (1.0 + math.exp((p - adp_val)/scale_val)) for p in range(1, league.teams*rounds+1) ]
    else:
        sigma_val = float(sel_row["Sigma"].values[0])
        curve = [ 1.0 - normal_cdf((p - adp_val)/sigma_val) for p in range(1, league.teams*rounds+1) ]
    st.line_chart(pd.DataFrame({"Pick": list(range(1, league.teams*rounds+1)), "PAvail": curve}).set_index("Pick"))

# -----------------------------
# SnakeSim
# -----------------------------
st.subheader("SnakeSim (others pick by ADP; you pick by PreDraft WeeklyEV)")
snake_df = (pd.DataFrame({"sum_wev":[]}) if len(model_df)==0 else
            pd.DataFrame({"sum_wev":[0.0]}) )
snake_df = snake_df.iloc[0:0]
if len(model_df)>0:
    def snake_sim(model_df: pd.DataFrame, league, var, rounds_sim=8, runs=200) -> pd.DataFrame:
        df = model_df.copy().sort_values("ADP").reset_index(drop=True)
        N = len(df)
        res = []
        for run in range(runs):
            if var.use_adp_sigma and "Sigma" in df.columns:
                sigma = df["Sigma"].values
            elif var.use_linear_sigma:
                sigma = np.clip(var.sigma_a + var.sigma_b * df["ADP"].values, var.sigma_min, var.sigma_max)
            else:
                sigma = np.full(N, var.sigma_const)
            draw = df["ADP"].values + np.random.normal(0, sigma)
            sim_rank = np.argsort(np.argsort(draw))
            taken = np.zeros(N, dtype=bool)
            my_picks = []
            for r in range(1, int(rounds_sim)+1):
                pick = pick_number(r, league.teams, league.slot) - 1
                for idx in np.where(sim_rank < pick)[0]:
                    taken[idx] = True
                avail_idx = np.where(~taken)[0]
                if avail_idx.size == 0: break
                best_idx = avail_idx[np.argmax(df.iloc[avail_idx]["PreWeeklyEV_base"].values)]
                taken[best_idx] = True
                my_picks.append(best_idx)
            sum_wev = float(df.iloc[my_picks]["PreWeeklyEV_base"].sum()) if my_picks else 0.0
            res.append({"sum_wev": sum_wev, "num_picks": len(my_picks)})
        return pd.DataFrame(res)

    snake_df = snake_sim(model_df, league, var, rounds_sim=int(rounds_sim), runs=int(runs_sim))
    st.write(f"Runs: {len(snake_df)} | Mean sum WeeklyEV over first {int(rounds_sim)} rounds: {snake_df['sum_wev'].mean():.1f}")
    st.bar_chart(snake_df["sum_wev"])

# -----------------------------
# Exports
# -----------------------------
st.subheader("Exports")
csv = targets.to_csv(index=False).encode("utf-8")
st.download_button("Download targets (CSV)", csv, file_name="targets_round.csv", mime="text/csv")
state = {"taken": list(st.session_state["taken"]), "mine": list(st.session_state["mine"])}
st.download_button("Download flags (JSON)", json.dumps(state).encode("utf-8"), file_name="draft_state.json", mime="application/json")

st.caption("ADP sigma is used when present. Planner uses a beam search to account for roster interaction and scarcity across early rounds.")
