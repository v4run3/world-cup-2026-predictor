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
    "https://raw.githubusercontent.com/martj42/"
    "international_results/master/results.csv"
)

# Raw CSV columns we expect.
_RAW_COLUMNS = [
    "date", "home_team", "away_team",
    "home_score", "away_score", "tournament", "city", "country", "neutral",
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
    print(f"Played matches : {len(played):,}  "
          f"({played.date.min().date()} -> {played.date.max().date()})")
    print(f"World Cups      : {(played.tournament == 'FIFA World Cup').sum():,} matches")
    print(f"2026 fixtures   : {len(fixtures)}")
    print(f"2026 teams      : {len(teams)}")
    print(", ".join(teams))


if __name__ == "__main__":
    if "--download" in sys.argv:
        download_results()
    _summary()
