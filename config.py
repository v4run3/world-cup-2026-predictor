"""Central configuration: paths, constants, and (optional) API keys.

API keys are read from environment variables so nothing secret is committed.
Set them in a local .env file or your shell before running, e.g.:
    setx FOOTBALL_DATA_API_KEY "your-key"   (Windows)
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"

# Files produced by the pipeline (created in later steps)
RESULTS_CSV = RAW_DIR / "results.csv"  # historical match results
RANKINGS_CSV = RAW_DIR / "fifa_rankings.csv"  # FIFA ranking history
FEATURES_CSV = PROCESSED_DIR / "features.csv"  # engineered training table
XGB_MODEL = MODELS_DIR / "xgb_outcome.json"  # win/draw/loss classifier
POISSON_MODEL = MODELS_DIR / "poisson_scoreline.pkl"
ELO_RATINGS = PROCESSED_DIR / "elo_ratings.csv"
TEAM_STATE_CSV = PROCESSED_DIR / "team_state.csv"  # latest elo + form per team

# --- API keys (optional, for live 2026 data) -------------------------------
FOOTBALL_DATA_API_KEY = os.getenv("FOOTBALL_DATA_API_KEY", "")

# --- Model / Elo constants -------------------------------------------------
ELO_BASE = 1500  # starting rating for every team
ELO_K = 30  # update sensitivity per match
ELO_HOME_ADVANTAGE = 65  # rating points added to the home side

# Outcome label encoding used everywhere
OUTCOME_LABELS = {0: "Away Win", 1: "Draw", 2: "Home Win"}

# Feature engineering / training
TRAIN_START_YEAR = 1990  # Elo uses full history; model trains on matches >= this
FORM_WINDOW = 10  # how many recent matches define a team's "form"

# Placeholder list of qualified / expected 2026 nations (edit as draw finalises)
WORLD_CUP_2026_TEAMS = [
    "Argentina",
    "France",
    "Brazil",
    "England",
    "Spain",
    "Portugal",
    "Netherlands",
    "Germany",
    "Belgium",
    "Croatia",
    "Uruguay",
    "USA",
    "Mexico",
    "Canada",
    "Morocco",
    "Japan",
    "Senegal",
    "Colombia",
]

H2H_CSV = PROCESSED_DIR / "h2h.csv"  # head-to-head lookup table
H2H_WINDOW = 5  # last N direct matchups to consider

# Tournament K-factors for Elo updates — higher means WC matches shift ratings more.
# Based on World Football Elo Ratings methodology.
TOURNAMENT_K_FACTORS = {
    "FIFA World Cup": 60,
    "UEFA European Championship": 50,
    "Copa América": 50,
    "Africa Cup of Nations": 50,
    "AFC Asian Cup": 45,
    "CONCACAF Gold Cup": 40,
    "OFC Nations Cup": 40,
    "FIFA Confederations Cup": 50,
    "UEFA Nations League": 35,
    "CONCACAF Nations League": 35,
}
ELO_K_DEFAULT = 30  # default for unlisted competitions
ELO_K_FRIENDLY = 20  # any tournament name containing "friendly"
