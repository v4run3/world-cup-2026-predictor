"""Step 8 — Self-bootstrap so the app works on a fresh clone / cloud deploy.

The data CSVs and model artifacts are git-ignored (reproducible, not committed).
On first launch this module downloads the data and builds everything if missing,
so the deployed app is fully functional without any manual setup.
"""

from __future__ import annotations

import config


def _need(*paths) -> bool:
    return any(not p.exists() for p in paths)


def ensure_ready(log=print) -> None:
    """Make sure raw data, processed features, and models all exist."""
    # 1. Raw historical results.
    if _need(config.RESULTS_CSV):
        log("Downloading historical match data...")
        from src.data_collection import download_results

        download_results()

    # 2. Processed features / Elo / team state / H2H.
    if _need(
        config.FEATURES_CSV, config.ELO_RATINGS, config.TEAM_STATE_CSV, config.H2H_CSV
    ):
        log("Building features, Elo ratings, and H2H table...")
        from src.data_collection import load_results
        from src.features import (
            build_features,
            build_h2h_table,
            current_elo_ratings,
            current_team_state,
        )

        res = load_results(played_only=True)
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        build_features(res).to_csv(config.FEATURES_CSV, index=False)
        current_elo_ratings(res).to_csv(config.ELO_RATINGS, index=False)
        current_team_state(res).to_csv(config.TEAM_STATE_CSV, index=False)
        build_h2h_table(res).to_csv(config.H2H_CSV, index=False)

    # 3. Trained models.
    if _need(config.XGB_MODEL, config.POISSON_MODEL):
        log("Training models (one-time, ~30s)...")
        from src.model import train_outcome_model, train_scoreline_model

        train_outcome_model(verbose=False)
        train_scoreline_model(verbose=False)

    log("Ready.")
