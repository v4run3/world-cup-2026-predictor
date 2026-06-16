"""Step 3 — Feature engineering.

Turns raw match results into a model-ready table. Key ideas:

1. **Elo** is computed chronologically over the *entire* history so every team's
   strength is well calibrated. Variable K-factors weight important matches
   (World Cup K=60) more than friendlies (K=20), matching real football Elo
   methodology. Pre-match ratings are recorded as features; post-match ratings
   are stored for the next match.

2. **Recent form** (rolling points & goals over the last N matches) captures
   short-term momentum that Elo smooths over.

3. **Head-to-head** records capture psychological/stylistic matchup edges.

4. **Tournament importance** is encoded as both a training sample weight and
   a direct feature, so the model learns that WC matches are higher stakes.

The final training table is filtered to matches from config.TRAIN_START_YEAR
onward (Elo still uses the full pre-1990 history for calibration).

Run directly to build and save everything:
    python -m src.features
"""

from __future__ import annotations

from collections import defaultdict, deque

import pandas as pd

import config
from src.data_collection import load_results

# ---------------------------------------------------------------------------
# Tournament K-factor helpers
# ---------------------------------------------------------------------------


def tournament_k_factor(tournament: str) -> float:
    """Return the Elo K-factor appropriate for this tournament type."""
    t_lower = str(tournament).lower()
    if "friendly" in t_lower:
        return config.ELO_K_FRIENDLY
    for name, k in config.TOURNAMENT_K_FACTORS.items():
        if name.lower() in t_lower:
            return k
    # Keyword fallback
    if "world cup" in t_lower:
        return 40 if "qualif" in t_lower else 60
    if any(
        x in t_lower
        for x in ["euro", "copa", "africa", "asian", "gold cup", "confederation"]
    ):
        return 45 if "qualif" in t_lower else 50
    if "qualif" in t_lower or "qualifier" in t_lower:
        return 35
    return config.ELO_K_DEFAULT


def tournament_importance(tournament: str) -> float:
    """Normalised importance weight for training sample weighting (0.33–1.0)."""
    k = tournament_k_factor(tournament)
    return round(k / 60.0, 3)  # 1.0 for WC, ~0.33 for friendlies


# ---------------------------------------------------------------------------
# Elo primitives
# ---------------------------------------------------------------------------


def expected_score(rating_a: float, rating_b: float) -> float:
    """Elo expected score (win probability) for team A vs team B."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(
    rating_a: float, rating_b: float, score_a: float, k: float = config.ELO_K
) -> tuple[float, float]:
    """Return updated (rating_a, rating_b) after a match.

    score_a is the actual result for A: 1 win, 0.5 draw, 0 loss.
    k is the K-factor (sensitivity); higher k = bigger rating shifts.
    """
    exp_a = expected_score(rating_a, rating_b)
    change = k * (score_a - exp_a)
    return rating_a + change, rating_b - change


def _outcome_label(home_score: int, away_score: int) -> int:
    """Map a result to the shared encoding: 2 home win, 1 draw, 0 away win."""
    if home_score > away_score:
        return 2
    if home_score == away_score:
        return 1
    return 0


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------


def build_features(results: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the per-match feature table (filtered to TRAIN_START_YEAR+).

    Walks every match in date order, emitting pre-match features, then updates
    the rolling Elo / form / H2H state with the actual result.
    """
    if results is None:
        results = load_results(played_only=True)
    results = results.sort_values("date").reset_index(drop=True)

    elo: dict[str, float] = defaultdict(lambda: config.ELO_BASE)
    form: dict[str, deque] = defaultdict(lambda: deque(maxlen=config.FORM_WINDOW))
    # H2H: frozenset({teamA, teamB}) -> deque of (home_team, home_score, away_score)
    h2h: dict[frozenset, deque] = defaultdict(lambda: deque(maxlen=config.H2H_WINDOW))

    rows = []
    for r in results.itertuples(index=False):
        home, away = str(r.home_team), str(r.away_team)
        eh, ea = elo[home], elo[away]
        neutral = bool(r.neutral)
        k = tournament_k_factor(r.tournament)
        t_weight = tournament_importance(r.tournament)

        # Home advantage only applies at non-neutral venues.
        adv = 0 if neutral else config.ELO_HOME_ADVANTAGE
        exp_home = expected_score(eh + adv, ea)

        hf, af = form[home], form[away]

        # --- Head-to-head stats (pre-match, using past encounters only) ---
        pair = frozenset([home, away])
        h2h_hist = list(h2h[pair])
        h2h_home_wins = sum(
            1
            for ht, hs, as_ in h2h_hist
            if (ht == home and hs > as_) or (ht == away and as_ > hs)
        )
        h2h_away_wins = sum(
            1
            for ht, hs, as_ in h2h_hist
            if (ht == away and hs > as_) or (ht == home and as_ > hs)
        )
        total_h2h = len(h2h_hist)
        h2h_home_adv = (
            (h2h_home_wins - h2h_away_wins) / total_h2h if total_h2h > 0 else 0.0
        )

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
            "home_form_gd": round(_avg(hf, 0) - _avg(hf, 1), 3),
            "away_form_gd": round(_avg(af, 0) - _avg(af, 1), 3),
            "tournament_weight": t_weight,
            "h2h_home_adv": round(h2h_home_adv, 3),
            "home_goals": int(r.home_score),
            "away_goals": int(r.away_score),
            "target": _outcome_label(r.home_score, r.away_score),
        }
        rows.append(row)

        # --- Update state with actual result ---
        score_home = (
            1.0
            if r.home_score > r.away_score
            else (0.5 if r.home_score == r.away_score else 0.0)
        )
        new_eh, new_ea = update_elo(eh + adv, ea, score_home, k=k)
        elo[home] = new_eh - adv
        elo[away] = new_ea
        hf.append((r.home_score, r.away_score, _points(r.home_score, r.away_score)))
        af.append((r.away_score, r.home_score, _points(r.away_score, r.home_score)))
        h2h[pair].append((home, r.home_score, r.away_score))

    feats = pd.DataFrame(rows)
    feats = feats[feats["date"].dt.year >= config.TRAIN_START_YEAR].reset_index(
        drop=True
    )
    return feats


# ---------------------------------------------------------------------------
# Current team state (used at prediction time)
# ---------------------------------------------------------------------------


def current_team_state(results: pd.DataFrame | None = None) -> pd.DataFrame:
    """Each team's latest Elo + rolling form, as of their most recent match.

    These are exactly the inputs needed to predict a *future* match.
    """
    if results is None:
        results = load_results(played_only=True)
    results = results.sort_values("date").reset_index(drop=True)

    elo: dict[str, float] = defaultdict(lambda: config.ELO_BASE)
    form: dict[str, deque] = defaultdict(lambda: deque(maxlen=config.FORM_WINDOW))

    for r in results.itertuples(index=False):
        home, away = str(r.home_team), str(r.away_team)
        adv = 0 if bool(r.neutral) else config.ELO_HOME_ADVANTAGE
        k = tournament_k_factor(r.tournament)
        score_home = (
            1.0
            if r.home_score > r.away_score
            else (0.5 if r.home_score == r.away_score else 0.0)
        )
        nh, na = update_elo(elo[home] + adv, elo[away], score_home, k=k)
        elo[home], elo[away] = nh - adv, na
        form[home].append(
            (r.home_score, r.away_score, _points(r.home_score, r.away_score))
        )
        form[away].append(
            (r.away_score, r.home_score, _points(r.away_score, r.home_score))
        )

    rows = []
    for team, rating in elo.items():
        hist = form[team]
        rows.append(
            {
                "team": team,
                "elo": rating,
                "form_pts": _avg(hist, 2),
                "gf_avg": _avg(hist, 0),
                "ga_avg": _avg(hist, 1),
                "form_gd": round(_avg(hist, 0) - _avg(hist, 1), 3),
            }
        )
    return pd.DataFrame(rows).sort_values("elo", ascending=False).reset_index(drop=True)


def load_team_state() -> dict[str, dict]:
    """Load saved per-team state as {team: {elo, form_pts, gf_avg, ga_avg, form_gd}}."""
    if config.TEAM_STATE_CSV.exists():
        df = pd.read_csv(config.TEAM_STATE_CSV)
        return df.set_index("team").to_dict("index")
    return {}


# ---------------------------------------------------------------------------
# Elo ratings
# ---------------------------------------------------------------------------


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
        home, away = str(r.home_team), str(r.away_team)
        adv = 0 if bool(r.neutral) else config.ELO_HOME_ADVANTAGE
        k = tournament_k_factor(r.tournament)
        score_home = (
            1.0
            if r.home_score > r.away_score
            else (0.5 if r.home_score == r.away_score else 0.0)
        )
        nh, na = update_elo(elo[home] + adv, elo[away], score_home, k=k)
        elo[home], elo[away] = nh - adv, na

    return (
        pd.DataFrame({"team": list(elo.keys()), "elo": list(elo.values())})
        .sort_values("elo", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Head-to-head table
# ---------------------------------------------------------------------------


def build_h2h_table(results: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build a head-to-head advantage table for all team pairs.

    For each unordered pair {A, B}, computes h2h_a_adv = (A_wins - B_wins) /
    total_matches, where A is arbitrarily the first team alphabetically.
    Range is [-1, +1]; 0 means equal record or no history.
    """
    if results is None:
        results = load_results(played_only=True)
    results = results.sort_values("date").reset_index(drop=True)

    h2h: dict[frozenset, deque] = defaultdict(lambda: deque(maxlen=config.H2H_WINDOW))
    for r in results.itertuples(index=False):
        pair = frozenset([str(r.home_team), str(r.away_team)])
        h2h[pair].append((str(r.home_team), r.home_score, r.away_score))

    rows = []
    for pair, history in h2h.items():
        teams = sorted(pair)
        if len(teams) != 2:
            continue
        t1, t2 = teams[0], teams[1]
        t1_wins = sum(
            1
            for ht, hs, as_ in history
            if (ht == t1 and hs > as_) or (ht == t2 and as_ > hs)
        )
        t2_wins = sum(
            1
            for ht, hs, as_ in history
            if (ht == t2 and hs > as_) or (ht == t1 and as_ > hs)
        )
        total = len(history)
        h2h_adv = (t1_wins - t2_wins) / max(total, 1)
        rows.append({"team_a": t1, "team_b": t2, "h2h_a_adv": round(h2h_adv, 3)})

    return pd.DataFrame(rows)


def load_h2h() -> dict[tuple[str, str], float]:
    """Load H2H table as {(team_a, team_b): advantage_for_a} or {} if missing."""
    if config.H2H_CSV.exists():
        df = pd.read_csv(config.H2H_CSV)
        return {
            (str(row["team_a"]), str(row["team_b"])): float(row["h2h_a_adv"])
            for _, row in df.iterrows()
        }
    return {}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


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
    h2h_table = build_h2h_table(res)
    h2h_table.to_csv(config.H2H_CSV, index=False)
    print(f"Wrote {len(feats):,} feature rows -> {config.FEATURES_CSV}")
    print(f"Wrote {len(ratings):,} Elo ratings -> {config.ELO_RATINGS}")
    print(f"Wrote {len(state):,} team states -> {config.TEAM_STATE_CSV}")
    print(f"Wrote {len(h2h_table):,} H2H pairs -> {config.H2H_CSV}")
    print("\nTop 15 teams by Elo:")
    print(ratings.head(15).to_string(index=False))
