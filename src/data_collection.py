"""Step 2 — Data collection.

Loads raw historical international match data from data/raw/results.csv and
splits it into:
  * played matches  -> training data (scores present)
  * 2026 fixtures   -> the real World Cup 2026 schedule (scores are NA)

Data source (free, no login): the GitHub mirror of the Kaggle dataset
"International football results from 1872 to present"
    https://raw.githubusercontent.com/martj42/international_results/master/results.csv

Download it once with:
    python -m src.data_collection --download
or just run this module to verify what's on disk.
"""

from __future__ import annotations

import sys
import urllib.request

import pandas as pd

import config

RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

# Raw CSV columns we expect.
_RAW_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
    "neutral",
]


def download_results() -> None:
    """Download the latest results.csv into data/raw/."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {RESULTS_URL} ...")
    urllib.request.urlretrieve(RESULTS_URL, config.RESULTS_CSV)
    print(f"Saved to {config.RESULTS_CSV}")


def _read_raw() -> pd.DataFrame:
    """Read the raw CSV with correct encoding and typed columns."""
    if not config.RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"{config.RESULTS_CSV} not found. Run: python -m src.data_collection --download"
        )
    df = pd.read_csv(
        config.RESULTS_CSV,
        encoding="utf-8",
        parse_dates=["date"],
        dtype={"home_team": "string", "away_team": "string"},
    )
    # neutral comes in as "TRUE"/"FALSE" strings.
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")
    return df


def load_results(played_only: bool = True) -> pd.DataFrame:
    """Return historical match results, sorted by date.

    played_only=True drops future fixtures (rows with no score yet).
    """
    df = _read_raw()
    if played_only:
        df = df[df["home_score"].notna() & df["away_score"].notna()].copy()
        df["home_score"] = df["home_score"].astype(int)
        df["away_score"] = df["away_score"].astype(int)
    return df.sort_values("date").reset_index(drop=True)


def load_fixtures_2026() -> pd.DataFrame:
    """Return the scheduled World Cup 2026 fixtures (scores not yet played)."""
    df = _read_raw()
    fut = df[df["home_score"].isna()].copy()
    fut = fut[fut["tournament"].str.contains("World Cup", case=False, na=False)]
    return fut.sort_values("date").reset_index(drop=True)


def get_world_cup_2026_teams() -> list[str]:
    """Return the sorted list of nations appearing in the 2026 fixtures."""
    fix = load_fixtures_2026()
    teams = set(fix["home_team"]) | set(fix["away_team"])
    return sorted(str(t) for t in teams)


def world_cup_2026_teams_safe() -> list[str]:
    """Teams from the dataset, falling back to the static config list.

    Lets the UI work even before results.csv has been downloaded.
    """
    try:
        teams = get_world_cup_2026_teams()
        return teams or list(config.WORLD_CUP_2026_TEAMS)
    except FileNotFoundError:
        return list(config.WORLD_CUP_2026_TEAMS)


def _summary() -> None:
    """Print a quick sanity report (used when running the module directly)."""
    played = load_results(played_only=True)
    fixtures = load_fixtures_2026()
    teams = get_world_cup_2026_teams()
    print(
        f"Played matches : {len(played):,}  "
        f"({played.date.min().date()} -> {played.date.max().date()})"
    )
    print(
        f"World Cups      : {(played.tournament == 'FIFA World Cup').sum():,} matches"
    )
    print(f"2026 fixtures   : {len(fixtures)}")
    print(f"2026 teams      : {len(teams)}")
    print(", ".join(teams))


# Path to the manual results patch file (committed to git, updated as results come in
# before the upstream dataset is refreshed).
_WC2026_PATCH_CSV = config.RAW_DIR / "wc2026_patch.csv"


def _apply_wc2026_patch(df: pd.DataFrame) -> pd.DataFrame:
    """Overlay confirmed results from the local patch file onto the fixture table.

    The patch file (data/raw/wc2026_patch.csv) holds matches whose scores are
    known but not yet pushed to the upstream dataset. Each patch row is matched
    by (date, home_team, away_team) and the scores are updated in-place.
    """
    if not _WC2026_PATCH_CSV.exists():
        return df

    patch = pd.read_csv(_WC2026_PATCH_CSV, parse_dates=["date"])
    patch["neutral"] = patch["neutral"].astype(str).str.upper().eq("TRUE")

    df = df.copy()
    for _, p in patch.iterrows():
        mask = (
            (df["date"].dt.date == p["date"].date())
            & (df["home_team"] == p["home_team"])
            & (df["away_team"] == p["away_team"])
        )
        if mask.any():
            df.loc[mask, "home_score"] = float(p["home_score"])
            df.loc[mask, "away_score"] = float(p["away_score"])

    return df


def load_all_wc2026_matches() -> pd.DataFrame:
    """Return ALL FIFA World Cup 2026 matches: played (with scores) and upcoming (NaN scores).

    Filters for the exact tournament name "FIFA World Cup" to exclude qualifiers.
    Applies a local patch file (data/raw/wc2026_patch.csv) so that results
    confirmed before the upstream dataset is refreshed are shown immediately.
    Adds a boolean column `is_played` to distinguish played from upcoming matches.
    """
    df = _read_raw()
    wc2026 = df[
        (df["tournament"] == "FIFA World Cup") & (df["date"].dt.year >= 2026)
    ].copy()

    # Overlay any locally-known results not yet in the upstream CSV.
    wc2026 = _apply_wc2026_patch(wc2026)

    wc2026["is_played"] = wc2026["home_score"].notna()
    # Cast scores to nullable Int64 so played matches show integers, upcoming show NaN.
    wc2026["home_score"] = pd.to_numeric(wc2026["home_score"], errors="coerce").astype(
        "Int64"
    )
    wc2026["away_score"] = pd.to_numeric(wc2026["away_score"], errors="coerce").astype(
        "Int64"
    )
    return wc2026.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Full-tournament schedule including knockout placeholders
# ---------------------------------------------------------------------------

# Official knockout-stage schedule. Teams are TBD until groups are decided.
# Venues use the same city names as the rest of the dataset.
_KNOCKOUT_SCHEDULE = [
    # (date, round_name, city, country)
    # Round of 32 — 16 matches, Jun 28 – Jul 3
    ("2026-06-28", "Round of 32", "Inglewood", "United States"),
    ("2026-06-29", "Round of 32", "Foxborough", "United States"),
    ("2026-06-29", "Round of 32", "Guadalupe", "Mexico"),
    ("2026-06-29", "Round of 32", "Houston", "United States"),
    ("2026-06-30", "Round of 32", "East Rutherford", "United States"),
    ("2026-06-30", "Round of 32", "Arlington", "United States"),
    ("2026-06-30", "Round of 32", "Mexico City", "Mexico"),
    ("2026-07-01", "Round of 32", "Atlanta", "United States"),
    ("2026-07-01", "Round of 32", "Santa Clara", "United States"),
    ("2026-07-01", "Round of 32", "Seattle", "United States"),
    ("2026-07-02", "Round of 32", "Toronto", "Canada"),
    ("2026-07-02", "Round of 32", "Inglewood", "United States"),
    ("2026-07-02", "Round of 32", "Vancouver", "Canada"),
    ("2026-07-03", "Round of 32", "Miami Gardens", "United States"),
    ("2026-07-03", "Round of 32", "Kansas City", "United States"),
    ("2026-07-03", "Round of 32", "Arlington", "United States"),
    # Round of 16 — 8 matches, Jul 4 – Jul 7
    ("2026-07-04", "Round of 16", "Philadelphia", "United States"),
    ("2026-07-04", "Round of 16", "Houston", "United States"),
    ("2026-07-05", "Round of 16", "East Rutherford", "United States"),
    ("2026-07-05", "Round of 16", "Mexico City", "Mexico"),
    ("2026-07-06", "Round of 16", "Arlington", "United States"),
    ("2026-07-06", "Round of 16", "Seattle", "United States"),
    ("2026-07-07", "Round of 16", "Atlanta", "United States"),
    ("2026-07-07", "Round of 16", "Vancouver", "Canada"),
    # Quarter-finals — 4 matches, Jul 9 – Jul 12
    ("2026-07-09", "Quarter-final", "Foxborough", "United States"),
    ("2026-07-10", "Quarter-final", "Inglewood", "United States"),
    ("2026-07-11", "Quarter-final", "Miami Gardens", "United States"),
    ("2026-07-12", "Quarter-final", "Kansas City", "United States"),
    # Semi-finals — 2 matches, Jul 14 – Jul 15
    ("2026-07-14", "Semi-final", "Arlington", "United States"),
    ("2026-07-15", "Semi-final", "Atlanta", "United States"),
    # 3rd Place — Jul 18
    ("2026-07-18", "3rd Place", "Miami Gardens", "United States"),
    # Final — Jul 19
    ("2026-07-19", "Final", "East Rutherford", "United States"),
]


def load_knockout_placeholders() -> pd.DataFrame:
    """Return placeholder rows for all 32 knockout matches (teams are TBD).

    These are appended to the group-stage fixtures so the Schedule page can
    show the full tournament from Jun 11 through the Final on Jul 19.
    """
    rows = []
    for date_str, match_round, city, country in _KNOCKOUT_SCHEDULE:
        rows.append(
            {
                "date": pd.Timestamp(date_str),
                "home_team": "TBD",
                "away_team": "TBD",
                "home_score": pd.NA,
                "away_score": pd.NA,
                "tournament": "FIFA World Cup",
                "city": city,
                "country": country,
                "neutral": True,
                "is_played": False,
                "match_round": match_round,
            }
        )
    df = pd.DataFrame(rows)
    df["home_score"] = df["home_score"].astype("Int64")
    df["away_score"] = df["away_score"].astype("Int64")
    return df


if __name__ == "__main__":
    if "--download" in sys.argv:
        download_results()
    _summary()
