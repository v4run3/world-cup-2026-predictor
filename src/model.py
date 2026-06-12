"""Step 4 — Models.

Two models work together:
  * XGBoost multiclass classifier -> Away Win / Draw / Home Win + probabilities
  * Two Poisson regressors -> expected goals per side -> most likely scoreline

Training uses a **time-based split** (older matches train, most recent matches
test) — never a random split, which would leak future form into the past.

Run directly to train and evaluate:
    python -m src.model
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import config

# Features fed to the outcome classifier (must exist in features.csv).
OUTCOME_FEATURES = [
    "elo_home", "elo_away", "elo_diff", "elo_exp_home", "neutral",
    "home_form_pts", "away_form_pts",
    "home_gf_avg", "home_ga_avg", "away_gf_avg", "away_ga_avg",
]

# Features for the goal (Poisson) models.
GOAL_FEATURES = [
    "elo_home", "elo_away", "elo_diff", "neutral",
    "home_gf_avg", "away_ga_avg", "home_form_pts",
]

TEST_FRACTION = 0.15  # most-recent share held out for evaluation
MAX_GOALS = 8         # scoreline grid cap for the Poisson model


def _load_features() -> pd.DataFrame:
    if not config.FEATURES_CSV.exists():
        raise FileNotFoundError(
            f"{config.FEATURES_CSV} not found. Run: python -m src.features"
        )
    return pd.read_csv(config.FEATURES_CSV, parse_dates=["date"]).sort_values("date")


def _time_split(df: pd.DataFrame):
    cut = int(len(df) * (1 - TEST_FRACTION))
    return df.iloc[:cut], df.iloc[cut:]


def train_outcome_model(df: pd.DataFrame | None = None, verbose: bool = True):
    """Fit the XGBoost outcome classifier and save it. Returns the model."""
    if df is None:
        df = _load_features()
    train, test = _time_split(df)

    X_tr, y_tr = train[OUTCOME_FEATURES], train["target"]
    X_te, y_te = test[OUTCOME_FEATURES], test["target"]

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)

    if verbose:
        proba = model.predict_proba(X_te)
        preds = proba.argmax(axis=1)
        # Baseline: always predict the most common class (home win).
        baseline = accuracy_score(y_te, np.full(len(y_te), df["target"].mode()[0]))
        print("--- XGBoost outcome model ---")
        print(f"  train rows : {len(train):,}   test rows : {len(test):,}")
        print(f"  accuracy   : {accuracy_score(y_te, preds):.3f}  (majority baseline {baseline:.3f})")
        print(f"  log-loss   : {log_loss(y_te, proba):.3f}")
        print("  top features:")
        imp = sorted(zip(OUTCOME_FEATURES, model.feature_importances_), key=lambda x: -x[1])
        for name, val in imp[:5]:
            print(f"    {name:16s} {val:.3f}")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(config.XGB_MODEL)
    return model


def train_scoreline_model(df: pd.DataFrame | None = None, verbose: bool = True):
    """Fit two Poisson regressors (home/away goals) and save them."""
    if df is None:
        df = _load_features()
    train, test = _time_split(df)

    # Scale first: raw Elo (~1500-2100) overflows the Poisson log-link otherwise.
    home_model = make_pipeline(StandardScaler(), PoissonRegressor(alpha=0.1, max_iter=1000))
    away_model = make_pipeline(StandardScaler(), PoissonRegressor(alpha=0.1, max_iter=1000))
    home_model.fit(train[GOAL_FEATURES], _goals(train, "home"))
    away_model.fit(train[GOAL_FEATURES], _goals(train, "away"))

    if verbose:
        pred_h = home_model.predict(test[GOAL_FEATURES])
        pred_a = away_model.predict(test[GOAL_FEATURES])
        mae_h = np.abs(pred_h - _goals(test, "home")).mean()
        mae_a = np.abs(pred_a - _goals(test, "away")).mean()
        print("--- Poisson scoreline model ---")
        print(f"  goal MAE   : home {mae_h:.2f}  away {mae_a:.2f}")
        print(f"  avg xG     : home {pred_h.mean():.2f}  away {pred_a.mean():.2f}")

    bundle = {"home": home_model, "away": away_model, "features": GOAL_FEATURES}
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, config.POISSON_MODEL)
    return bundle


def most_likely_scoreline(lambda_home: float, lambda_away: float) -> tuple[int, int]:
    """Most probable (home, away) scoreline given expected goals."""
    h = poisson.pmf(np.arange(MAX_GOALS + 1), lambda_home)
    a = poisson.pmf(np.arange(MAX_GOALS + 1), lambda_away)
    grid = np.outer(h, a)  # grid[i, j] = P(home=i, away=j)
    i, j = np.unravel_index(grid.argmax(), grid.shape)
    return int(i), int(j)


def load_models():
    """Load saved (outcome_model, scoreline_bundle); (None, None) if untrained."""
    if not (config.XGB_MODEL.exists() and config.POISSON_MODEL.exists()):
        return None, None
    outcome = XGBClassifier()
    outcome.load_model(config.XGB_MODEL)
    scoreline = joblib.load(config.POISSON_MODEL)
    return outcome, scoreline


def _goals(df: pd.DataFrame, side: str) -> np.ndarray:
    """Actual goals scored by `side` ('home'/'away') — the Poisson target."""
    return df[f"{side}_goals"].to_numpy()


if __name__ == "__main__":
    feats = _load_features()
    train_outcome_model(feats)
    print()
    train_scoreline_model(feats)
