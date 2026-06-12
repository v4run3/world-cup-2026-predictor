"""Step 3 — Feature engineering.

Turns raw match results into a model-ready table. Two key ideas:

1. **Elo** is computed chronologically over the *entire* history so every team's
   strength is well calibrated (a win over any opponent counts). For each match
   we record the teams' Elo *before* kickoff — that's the feature; using the
   post-match rating would leak the result.

2. **Recent form** (rolling points & goals over the last N matches) captures
   short-term momentum that Elo smooths over.

The final training table is filtered to matches from config.TRAIN_START_YEAR
onward (Elo itself still benefits from the full pre-1990 history).

Run directly to build and save everything:
    python -m src.features
"""
from __future__ import annotations

from collections import defaultdict, deque

import pandas as pd

import config
from src.data_collection import load_results


# --- Elo primitives --------------------------------------------------------
def expected_score(rating_a: float, rating_b: float) -> float:
    """Elo expected score (win prob) for team A vs team B."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(rating_a: float, rating_b: float, score_a: float) -> tuple[float, float]:
    """Return updated (rating_a, rating_b) after a match.

    score_a is the actual result for A: 1 win, 0.5 draw, 0 loss.
    """
    exp_a = expected_score(rating_a, rating_b)
    change = config.ELO_K * (score_a - exp_a)
    return rating_a + change, rating_b - change


def _outcome_label(home_score: int, away_score: int) -> int:
    """Map a result to the shared encoding: 2 home win, 1 draw, 0 away win."""
    if home_score > away_score:
        return 2
    if home_score == away_score:
        return 1
    return 0


# --- Main builder ----------------------------------------------------------
def build_features(results: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the per-match feature table (filtered to TRAIN_START_YEAR+).

    Walks every match in date order, emitting pre-match features, then updates
    the rolling Elo / form state with the actual result.
    """
    if results is None:
        results = load_results(played_only=True)
    results = results.sort_values("date").reset_index(drop=True)

    elo: dict[str, float] = defaultdict(lambda: config.ELO_BASE)
    # per-team rolling history of (goals_for, goals_against, points)
    form: dict[str, deque] = defaultdict(lambda: deque(maxlen=config.FORM_WINDOW))

    rows = []
    for r in results.itertuples(index=False):
        home, away = r.home_team, r.away_team
        eh, ea = elo[home], elo[away]
        neutral = bool(r.neutral)

        # Home advantage only applies at non-neutral venues.
        adv = 0 if neutral else config.ELO_HOME_ADVANTAGE
        exp_home = expected_score(eh + adv, ea)

        hf, af = form[home], form[away]
        row = {
            "date": r.date,
            "home_team": home,
            "away_team": away,
            "tournament": r.tournament,
            "neutral": int(neutral),
            "elo_home": eh,
            "elo_away": ea,
            "elo_diff": eh - ea,
            "elo_exp_home": exp_home,
            "home_form_pts": _avg(hf, 2),
            "away_form_pts": _avg(af, 2),
            "home_gf_avg": _avg(hf, 0),
            "home_ga_avg": _avg(hf, 1),
            "away_gf_avg": _avg(af, 0),
            "away_ga_avg": _avg(af, 1),
            "home_goals": int(r.home_score),   # actual result — target for Poisson model
            "away_goals": int(r.away_score),
            "target": _outcome_label(r.home_score, r.away_score),
        }
        rows.append(row)

        # --- update state with the actual result ---
        score_home = 1.0 if r.home_score > r.away_score else (0.5 if r.home_score == r.away_score else 0.0)
        elo[home], elo[away] = update_elo(eh + adv, ea, score_home)
        # remove the advantage we temporarily added back out of the stored rating
        elo[home] -= adv
        hf.append((r.home_score, r.away_score, _points(r.home_score, r.away_score)))
        af.append((r.away_score, r.home_score, _points(r.away_score, r.home_score)))

    feats = pd.DataFrame(rows)
    feats = feats[feats["date"].dt.year >= config.TRAIN_START_YEAR].reset_index(drop=True)
    return feats


def current_team_state(results: pd.DataFrame | None = None) -> pd.DataFrame:
    """Each team's latest Elo + rolling form, as of their most recent match.

    These are exactly the inputs needed to predict a *future* match, so the
    prediction layer reads this table to build a feature vector.
    """
    if results is None:
        results = load_results(played_only=True)
    results = results.sort_values("date").reset_index(drop=True)

    elo: dict[str, float] = defaultdict(lambda: config.ELO_BASE)
    form: dict[str, deque] = defaultdict(lambda: deque(maxlen=config.FORM_WINDOW))

    for r in results.itertuples(index=False):
        adv = 0 if bool(r.neutral) else config.ELO_HOME_ADVANTAGE
        score_home = 1.0 if r.home_score > r.away_score else (0.5 if r.home_score == r.away_score else 0.0)
        nh, na = update_elo(elo[r.home_team] + adv, elo[r.away_team], score_home)
        elo[r.home_team], elo[r.away_team] = nh - adv, na
        form[r.home_team].append((r.home_score, r.away_score, _points(r.home_score, r.away_score)))
        form[r.away_team].append((r.away_score, r.home_score, _points(r.away_score, r.home_score)))

    rows = []
    for team, rating in elo.items():
        hist = form[team]
        rows.append({
            "team": team,
            "elo": rating,
            "form_pts": _avg(hist, 2),
            "gf_avg": _avg(hist, 0),
            "ga_avg": _avg(hist, 1),
        })
    return pd.DataFrame(rows).sort_values("elo", ascending=False).reset_index(drop=True)


def load_team_state() -> dict[str, dict]:
    """Load saved per-team state as {team: {elo, form_pts, gf_avg, ga_avg}}."""
    if config.TEAM_STATE_CSV.exists():
        df = pd.read_csv(config.TEAM_STATE_CSV)
        return df.set_index("team").to_dict("index")
    return {}


def load_elo_ratings() -> dict[str, float]:
    """Load saved {team: elo} ratings, or {} if not built yet."""
    if config.ELO_RATINGS.exists():
        df = pd.read_csv(config.ELO_RATINGS)
        return dict(zip(df["team"], df["elo"]))
    return {}


def current_elo_ratings(results: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return each team's final Elo rating after the full history."""
    if results is None:
        results = load_results(played_only=True)
    results = results.sort_values("date").reset_index(drop=True)

    elo: dict[str, float] = defaultdict(lambda: config.ELO_BASE)
    for r in results.itertuples(index=False):
        adv = 0 if bool(r.neutral) else config.ELO_HOME_ADVANTAGE
        score_home = 1.0 if r.home_score > r.away_score else (0.5 if r.home_score == r.away_score else 0.0)
        nh, na = update_elo(elo[r.home_team] + adv, elo[r.away_team], score_home)
        elo[r.home_team], elo[r.away_team] = nh - adv, na

    return (
        pd.DataFrame({"team": list(elo.keys()), "elo": list(elo.values())})
        .sort_values("elo", ascending=False)
        .reset_index(drop=True)
    )


# --- small helpers ---------------------------------------------------------
def _points(gf: int, ga: int) -> int:
    return 3 if gf > ga else (1 if gf == ga else 0)


def _avg(history: deque, idx: int) -> float:
    """Average of element `idx` across a team's recent-match history (0 if empty)."""
    if not history:
        return 0.0
    return round(sum(h[idx] for h in history) / len(history), 3)


if __name__ == "__main__":
    res = load_results(played_only=True)
    feats = build_features(res)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    feats.to_csv(config.FEATURES_CSV, index=False)
    ratings = current_elo_ratings(res)
    ratings.to_csv(config.ELO_RATINGS, index=False)
    state = current_team_state(res)
    state.to_csv(config.TEAM_STATE_CSV, index=False)
    print(f"Wrote {len(feats):,} feature rows -> {config.FEATURES_CSV}")
    print(f"Wrote {len(ratings):,} Elo ratings -> {config.ELO_RATINGS}")
    print(f"Wrote {len(state):,} team states -> {config.TEAM_STATE_CSV}")
    print("\nTop 15 teams by Elo:")
    print(ratings.head(15).to_string(index=False))
