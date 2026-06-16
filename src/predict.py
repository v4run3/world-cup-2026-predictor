"""Step 5 — Prediction layer.

The single entry point the Streamlit UI calls. It builds a feature vector for a
matchup from each team's current state (Elo + rolling form + H2H), runs the trained
XGBoost classifier for outcome probabilities and the Poisson model for the most
likely scoreline.

If the models or team-state table aren't built yet, it falls back to a
transparent Elo-only heuristic so the app still runs.

Return shape (stable contract the UI depends on):

    {
        "home_team", "away_team",
        "probs": {"home_win", "draw", "away_win"},   # sum to 1.0
        "scoreline": (home_goals, away_goals),
        "expected_goals": (lam_home, lam_away) | None,
        "strength": {"home": float, "away": float},   # 0-100
        "source": "model" | "heuristic",
    }
"""

from __future__ import annotations

import pandas as pd

import config
from src.features import expected_score, load_h2h, load_team_state
from src.model import (
    GOAL_FEATURES,
    OUTCOME_FEATURES,
    load_models,
    most_likely_scoreline,
)

_DEFAULT_ELO = 1500

# Lazily loaded + cached so a fresh deploy can bootstrap artifacts first.
_cache: dict = {}


def _ensure_loaded() -> None:
    if "team_state" in _cache:
        return
    from src.bootstrap import ensure_ready

    ensure_ready(log=lambda *_: None)
    _cache["team_state"] = load_team_state()
    _cache["models"] = load_models()
    _cache["h2h"] = load_h2h()


def _team_state() -> dict:
    _ensure_loaded()
    return _cache["team_state"]


def _models():
    _ensure_loaded()
    return _cache["models"]


def _h2h() -> dict:
    _ensure_loaded()
    return _cache.get("h2h", {})


def _h2h_lookup(home: str, away: str) -> float:
    """Return H2H advantage for home team: positive = home has won more recently."""
    h2h_data = _h2h()
    # Table stored with alphabetically-first team as team_a
    ta, tb = sorted([home, away])
    adv = h2h_data.get((ta, tb), 0.0)
    return adv if ta == home else -adv


def _state(team: str) -> dict:
    """Current state for a team, with neutral defaults for unknown teams."""
    return _team_state().get(
        team,
        {
            "elo": _DEFAULT_ELO,
            "form_pts": 1.0,
            "gf_avg": 1.0,
            "ga_avg": 1.0,
            "form_gd": 0.0,
        },
    )


def _to_strength_score(elo: float) -> float:
    """Map a raw Elo number to a friendly 0-100 strength score for the UI."""
    return round(max(0, min(100, (elo - 1500) / 7)), 1)


def _feature_row(
    home: str, away: str, neutral: bool, tournament_weight: float = 1.0
) -> dict:
    """Assemble the model feature vector for a single matchup."""
    h, a = _state(home), _state(away)
    adv = 0 if neutral else config.ELO_HOME_ADVANTAGE
    h2h_adv = _h2h_lookup(home, away)
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
        "home_form_gd": h.get("form_gd", round(h["gf_avg"] - h["ga_avg"], 3)),
        "away_form_gd": a.get("form_gd", round(a["gf_avg"] - a["ga_avg"], 3)),
        "tournament_weight": tournament_weight,
        "h2h_home_adv": h2h_adv,
    }


def match_lambdas(
    home_team: str, away_team: str, neutral: bool = True, tournament_weight: float = 1.0
) -> tuple[float, float]:
    """Expected goals (lambda_home, lambda_away) for a matchup.

    Used by the tournament Monte-Carlo. Falls back to an Elo-gap estimate if the
    Poisson model isn't available.
    """
    _, scoreline_model = _models()
    if scoreline_model is not None and bool(_team_state()):
        X = pd.DataFrame(
            [_feature_row(home_team, away_team, neutral, tournament_weight)]
        )
        lam_h = float(scoreline_model["home"].predict(X[GOAL_FEATURES])[0])
        lam_a = float(scoreline_model["away"].predict(X[GOAL_FEATURES])[0])
        return lam_h, lam_a
    # Fallback: derive crude expected goals from the Elo gap.
    h_elo, a_elo = _state(home_team)["elo"], _state(away_team)["elo"]
    gap = (h_elo + (0 if neutral else config.ELO_HOME_ADVANTAGE) - a_elo) / 200
    return max(0.2, 1.4 + 0.5 * gap), max(0.2, 1.4 - 0.5 * gap)


def predict_match(
    home_team: str, away_team: str, neutral: bool = True, tournament_weight: float = 1.0
) -> dict:
    """Predict a single match using the trained models (or a fallback heuristic)."""
    h_elo, a_elo = _state(home_team)["elo"], _state(away_team)["elo"]
    strength = {"home": _to_strength_score(h_elo), "away": _to_strength_score(a_elo)}

    outcome_model, scoreline_model = _models()
    models_ready = (
        outcome_model is not None
        and scoreline_model is not None
        and bool(_team_state())
    )

    if models_ready:
        row = _feature_row(home_team, away_team, neutral, tournament_weight)
        X = pd.DataFrame([row])

        # Outcome probabilities: model classes are 0=away, 1=draw, 2=home.
        proba = outcome_model.predict_proba(X[OUTCOME_FEATURES])[0]
        probs = {
            "away_win": round(float(proba[0]), 3),
            "draw": round(float(proba[1]), 3),
            "home_win": round(float(proba[2]), 3),
        }

        # Scoreline from the Poisson goal models.
        lam_h = float(scoreline_model["home"].predict(X[GOAL_FEATURES])[0])
        lam_a = float(scoreline_model["away"].predict(X[GOAL_FEATURES])[0])
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
