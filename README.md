# World Cup 2026 Predictor ⚽

A FIFA World Cup 2026 predictor built with **Streamlit**. It forecasts match
outcomes (win/draw/loss), win probabilities, scorelines, and team strength from
historical international football data.

## Models

| Output | Model |
|---|---|
| Win / Draw / Loss + probabilities | **XGBoost** multiclass classifier |
| Scoreline | **Poisson / Dixon-Coles** goal model |
| Team strength | **Elo** rating (also a model feature) |

## Data

- **Training:** Kaggle "International football results from 1872 to present"
  (`results.csv`) + FIFA ranking history. Place CSVs in `data/raw/`.
- **Live (optional):** [football-data.org](https://www.football-data.org) for
  2026 fixtures (set `FOOTBALL_DATA_API_KEY`).

## Project structure

```
app.py                  Streamlit home page
config.py               Paths, constants, API keys
pages/                  Match Predictor, Team Analysis, Tournament Sim
src/
  data_collection.py    Load results & rankings        (Step 2)
  features.py           Elo + feature engineering       (Step 3)
  model.py              XGBoost + Poisson training      (Step 4)
  predict.py            Prediction API used by the UI   (Step 5)
data/   models/         Generated data & model artifacts (git-ignored)
```

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501.

## Roadmap

1. ✅ Skeleton — runnable multi-page app
2. ✅ Data collection — 49k matches + real 2026 fixtures
3. ⬜ Feature engineering (Elo, form, goals)
4. ⬜ Model training (XGBoost + Poisson)
5. ⬜ Prediction layer
6. ⬜ Team analysis page
7. ⬜ Tournament simulation (Monte Carlo)
8. ⬜ Polish & deploy
