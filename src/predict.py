"""Step 5 — Prediction layer.

The single entry point the Streamlit UI calls. It builds a feature vector for a
matchup from each team's current state (Elo + rolling form), runs the trained
XGBoost classifier for outcome probabilities and the Poisson model for the most
likely scoreline.

If the models or team-state table aren't built yet, it falls back to a
transparent Elo-only heuristic so the app still runs.

Return shape (stable contract the UI depends on):

    {
        "home_team", "away_team",
        "probs": {"home_win", "draw", "away_win"},   # sum to 1.0
        "scoreline": (home_goals, away_goals),
        "strength": {"home": float, "away": float},   # 0-100
        "source": "model" | "heuristic",
    }
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.features import expected_score, load_team_state
from src.model import (
    GOAL_FEATURES,
    OUTCOME_FEATURES,
    load_models,
    most_likely_scoreline,
)

# Loaded once and cached at module level.
_TEAM_STATE = load_team_state()
_OUTCOME_MODEL, _SCORELINE_MODEL = load_models()

_DEFAULT_ELO = 1500


def _state(team: str) -> dict:
    """Current state for a team, with neutral defaults for unknown teams."""
    return _TEAM_STATE.get(
        team,
        {"elo": _DEFAULT_ELO, "form_pts": 1.0, "gf_avg": 1.0, "ga_avg": 1.0},
    )


def _to_strength_score(elo: float) -> float:
    """Map a raw Elo number to a friendly 0-100 strength score for the UI."""
    return round(max(0, min(100, (elo - 1500) / 7)), 1)


def _feature_row(home: str, away: str, neutral: bool) -> dict:
    """Assemble the model feature vector for a single matchup."""
    h, a = _state(home), _state(away)
    adv = 0 if neutral else config.ELO_HOME_ADVANTAGE
    return {
        "elo_home": h["elo"],
        "elo_away": a["elo"],
        "elo_diff": h["elo"] - a["elo"],
        "elo_exp_home": expected_score(h["elo"] + adv, a["elo"]),
        "neutral": int(neutral),
        "home_form_pts": h["form_pts"],
        "away_form_pts": a["form_pts"],
        "home_gf_avg": h["gf_avg"],
        "home_ga_avg": h["ga_avg"],
        "away_gf_avg": a["gf_avg"],
        "away_ga_avg": a["ga_avg"],
    }


def predict_match(home_team: str, away_team: str, neutral: bool = True) -> dict:
    """Predict a single match using the trained models (or a fallback heuristic)."""
    h_elo, a_elo = _state(home_team)["elo"], _state(away_team)["elo"]
    strength = {"home": _to_strength_score(h_elo), "away": _to_strength_score(a_elo)}

    models_ready = (
        _OUTCOME_MODEL is not None
        and _SCORELINE_MODEL is not None
        and bool(_TEAM_STATE)
    )

    if models_ready:
        row = _feature_row(home_team, away_team, neutral)
        X = pd.DataFrame([row])

        # Outcome probabilities: model classes are 0=away, 1=draw, 2=home.
        proba = _OUTCOME_MODEL.predict_proba(X[OUTCOME_FEATURES])[0]
        probs = {
            "away_win": round(float(proba[0]), 3),
            "draw": round(float(proba[1]), 3),
            "home_win": round(float(proba[2]), 3),
        }

        # Scoreline from the Poisson goal models.
        lam_h = float(_SCORELINE_MODEL["home"].predict(X[GOAL_FEATURES])[0])
        lam_a = float(_SCORELINE_MODEL["away"].predict(X[GOAL_FEATURES])[0])
        scoreline = most_likely_scoreline(lam_h, lam_a)

        return {
            "home_team": home_team,
            "away_team": away_team,
            "probs": probs,
            "scoreline": scoreline,
            "expected_goals": (round(lam_h, 2), round(lam_a, 2)),
            "strength": strength,
            "source": "model",
        }

    return _heuristic(home_team, away_team, neutral, h_elo, a_elo, strength)


def _heuristic(home_team, away_team, neutral, h_elo, a_elo, strength) -> dict:
    """Elo-only fallback used until the models/team-state are built."""
    adv = 0 if neutral else config.ELO_HOME_ADVANTAGE
    p_home_raw = expected_score(h_elo + adv, a_elo)
    draw = 0.26 - 0.15 * abs(p_home_raw - 0.5)
    home_win = p_home_raw * (1 - draw)
    away_win = (1 - p_home_raw) * (1 - draw)
    total = home_win + draw + away_win
    probs = {
        "home_win": round(home_win / total, 3),
        "draw": round(draw / total, 3),
        "away_win": round(away_win / total, 3),
    }
    gap = (h_elo - a_elo) / 200
    return {
        "home_team": home_team,
        "away_team": away_team,
        "probs": probs,
        "scoreline": (max(0, round(1.4 + 0.5 * gap)), max(0, round(1.4 - 0.5 * gap))),
        "expected_goals": None,
        "strength": strength,
        "source": "heuristic",
    }
