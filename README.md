# World Cup 2026 Predictor ⚽

A FIFA World Cup 2026 predictor built with **Streamlit**. It forecasts match
outcomes (win/draw/loss), win probabilities, scorelines, team strength, and
simulates the whole tournament to a champion — all from historical
international football data.

## Features

- 📅 **Schedule** — every fixture by matchday with predicted scoreline + win %.
- 📊 **Team Analysis** — Elo strength rating and rank for all 48 nations.
- 🏆 **Tournament Simulation** — Monte-Carlo the full bracket for title odds.
- 🥅 **Custom Matchup** — predict any hypothetical tie.

## Models

| Output | Model |
|---|---|
| Win / Draw / Loss + probabilities | **XGBoost** multiclass classifier (~60% accuracy) |
| Scoreline / expected goals | **Poisson** goal model |
| Team strength | **Variable-K Elo** rating (WC K=60, friendlies K=20) |
| Title odds | **Monte-Carlo** simulation over the Poisson model |

### Model improvements (v2)
- **Variable K-factor Elo** — World Cup matches (K=60) shift ratings 3× more than friendlies (K=20), matching real football Elo methodology.
- **Head-to-head feature** — recent direct matchup record between each pair (4th most important feature).
- **Goal-difference form** — captures attacking/defensive momentum beyond just points.
- **Tournament importance weighting** — XGBoost training gives WC matches 3× the weight of friendlies.
- **Symmetric Poisson features** — both home and away goal models now see the full feature set.
- **Actual scores on the Schedule page** — played WC 2026 matches show the real final score alongside the model's prediction.

## Data

- **Training:** "International football results from 1872 to present"
  (~49k matches) — auto-downloaded from the
  [GitHub mirror](https://github.com/martj42/international_results), no key needed.
- The dataset also contains the real **48-team field and group fixtures** for 2026.

## Project structure

```
app.py                  Streamlit home page
config.py               Paths, constants, API keys
pages/
  1_Schedule.py         Fixtures by day with predictions
  2_Team_Analysis.py    Elo strength ratings
  3_Tournament_Simulation.py   Monte-Carlo title odds
  4_Custom_Matchup.py   Any hypothetical fixture
src/
  bootstrap.py          Auto-download + build on first run
  data_collection.py    Load results & 2026 fixtures
  features.py           Elo + feature engineering
  model.py              XGBoost + Poisson training
  predict.py            Prediction API used by the UI
  simulate.py           Tournament Monte-Carlo engine
data/   models/         Generated artifacts (git-ignored, auto-built)
```

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
streamlit run app.py
```

On first launch the app **auto-downloads the data and trains the models**
(~30s, one time). The app then opens at http://localhost:8501.

### Keeping results current

The upstream dataset (`martj42/international_results`) may lag a day or two behind.
To patch in results that are confirmed but not yet in the dataset, add rows to
`data/raw/wc2026_patch.csv` (committed to git — same schema as `results.csv`):

```
date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
2026-06-15,Belgium,Egypt,1,1,FIFA World Cup,Seattle,United States,True
```

The Schedule page will pick these up instantly — no rebuild required.
When the upstream dataset eventually includes the result the patch entry is harmlessly overridden.

To (re)build artifacts manually:

```bash
python -m src.data_collection --download   # fetch latest results
python -m src.features                      # build features + Elo + team state
python -m src.model                         # train & evaluate models
```

## Deploy (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. At [share.streamlit.io](https://share.streamlit.io), create an app pointing
   at `app.py` on the `main` branch.
3. First boot runs the bootstrap (downloads data + trains models) automatically —
   no artifacts need to be committed.

## Roadmap

1. ✅ Skeleton — runnable multi-page app
2. ✅ Data collection — 49k matches + real 2026 fixtures
3. ✅ Feature engineering (Elo, form, goals)
4. ✅ Model training (XGBoost + Poisson)
5. ✅ Prediction layer
6. ✅ Team analysis page
7. ✅ Tournament simulation (Monte Carlo)
8. ✅ Polish & deploy
9. ✅ Actual scores on the Schedule page (live WC 2026 results)
10. ✅ Model refinement — variable K-factor Elo, H2H features, tournament weighting
