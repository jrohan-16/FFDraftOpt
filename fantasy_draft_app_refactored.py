# -*- coding: utf-8 -*-
"""
Fantasy Draft Console — Streamlit UI (refactored)
- UI-only: all ingestion, modeling, and recommendation logic lives in fantasy_draft_engine.py
- Upload-only workflow preserved
Run:
    streamlit run fantasy_draft_app_refactored.py
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

from fantasy_draft_engine import (
    LeagueConfig, VarianceModel, InjuryModel,
    load_fp_uploads, load_adp_upload,
    build_model_df, compute_targets, pick_number
)

# =============================================================================
# UI
# =============================================================================

st.set_page_config(page_title="Fantasy Draft Console (Refactored)", layout="centered")
st.title("Draft Console (Refactored)")

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

# Cached wrappers (UI-only) ---------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_proj_cached(qb_b: bytes|None, flx_b: bytes|None, k_b: bytes|None, dst_b: bytes|None) -> pd.DataFrame:
    return load_fp_uploads(qb_b, flx_b, k_b, dst_b)

@st.cache_data(show_spinner=False)
def _load_adp_cached(adp_b: bytes|None) -> pd.DataFrame:
    return load_adp_upload(adp_b)

@st.cache_data(show_spinner=False)
def _build_model_cached(proj: pd.DataFrame, league_: LeagueConfig, var_: VarianceModel, inj_: InjuryModel,
                        risk_map: dict, cv_map: dict, adp_: pd.DataFrame|None) -> pd.DataFrame:
    return build_model_df(proj, league_, var_, inj_, risk_map, cv_map, adp_df=adp_)

# Load uploads
proj_df = _load_proj_cached(
    qb_up.getvalue() if qb_up else None,
    flx_up.getvalue() if flx_up else None,
    k_up.getvalue() if k_up else None,
    dst_up.getvalue() if dst_up else None,
)
adp_df = _load_adp_cached(adp_up.getvalue() if adp_up else None)

if proj_df is None or proj_df.empty:
    st.info("📥 Upload at least the FLX projections (RB/WR/TE). QB/K/DST are optional but recommended.")
    st.stop()

# Build model dataframe
model_df = _build_model_cached(proj_df, league, var, inj, risk_by_pos, cv_by_pos, adp_df)
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

# Compute targets (engine call)
unavailable = st.session_state["taken"].union(st.session_state["mine"])
targets = compute_targets(
    model_df, round_now, league, var, top_n=topN,
    unavailable_pids=unavailable, risk_lambda=float(st.session_state.get("risk_lambda", 0.15))
)

# Render
st.markdown("### Recommended now")
total_avail = int((~model_df["PID"].isin(unavailable)).sum())
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
