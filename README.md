
# Fantasy Draft Optimizer — Streamlit MVP v1.2

## New in v1.2
- **ADP integration**: auto-loads the uploaded `FantasyPros_2025_Overall_ADP_Rankings.csv` (or any ADP/ECR CSV) and joins to projections.
- **Per-player σ from ADP**: if the ADP file includes *Std Dev* or *Min/Max*, we derive a per-player sigma and use it for **P(Avail)**.
- **Early-Round Planner (Beam Search)**: plans first K rounds with roster-aware marginal value and \(P(Avail)\). Controls for pool size, beam width, excluding K/DST early, and QB caps.
- **Calibration view**: visualize \(P(Avail)\) curve for any player.
- **Minor**: more robust CSV parsing, cleaner UI, and exports.

## Run
```bash
pip install -r requirements.txt
streamlit run fantasy_draft_app.py
```

## Tips
- Upload your FantasyPros QB/FLX/K/DST projection CSVs, then upload the ADP CSV.
- Turn on **Use logistic P(Avail)** if your room has heavier tails than Normal.
- In **Planner**, start with: rounds=8, pool=12, beam=60, exclude K/DST < round 10, max 1 QB.
