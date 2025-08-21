# Refactor: UI vs Logic

This refactor cleanly separates **UI** (Streamlit) from **logic** (pure Python).

## Files

- `fantasy_draft_engine.py` — *Pure engine*. No Streamlit. Handles ingestion, modeling, and recommendations.
- `fantasy_draft_app_refactored.py` — *UI-only* Streamlit app that imports the engine.
- Original monolithic file is unchanged.

## Run

```bash
streamlit run fantasy_draft_app_refactored.py
```

## What moved into the engine?

- CSV ingestion from uploaded **bytes** (`load_fp_uploads`, `load_adp_upload`)
- Modeling (`build_model_df`) including injury episode Markov approximation, FLEX-aware replacement, ADP sigma
- Draft math (`pick_number`, `tail_prob`) and recommendations (`compute_targets`)

## What stayed in the UI?

- Widgets, session state, caching decorators
- File uploaders and display
- Buttons (Draft / Mark taken / Hide) and reruns

## Notes

- Engine contains **no `streamlit` imports** and is importable by other front ends (FastAPI, CLI, etc.).
- The one previous Streamlit-specific reference inside the logic (`st.session_state` and `@st.cache_data`) is removed. The engine now takes explicit parameters (e.g., `unavailable_pids`, `risk_lambda`).

