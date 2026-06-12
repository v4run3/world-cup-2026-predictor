"""Schedule page — all World Cup 2026 fixtures by day, each with a prediction.

This is the primary view: pick a matchday and see every game that day with its
predicted scoreline and win probabilities. The full knockout bracket arrives
with the tournament simulation (Step 7), once group results are simulated.
"""
import pandas as pd
import streamlit as st

import src.data_collection as dc
from src.predict import predict_match

st.set_page_config(page_title="Schedule", page_icon="📅", layout="wide")
st.title("📅 Match Schedule & Predictions")


@st.cache_data
def get_fixtures() -> pd.DataFrame:
    return dc.load_fixtures_2026()


@st.cache_data
def predict_fixture(home: str, away: str, neutral: bool) -> dict:
    return predict_match(home, away, neutral=neutral)


fixtures = get_fixtures()

if fixtures.empty:
    st.warning("No fixtures found. Run `python -m src.data_collection --download`.")
    st.stop()

# If the trained models aren't available, predictions fall back to a heuristic.
sample = predict_fixture(fixtures.iloc[0].home_team, fixtures.iloc[0].away_team, True)
if sample["source"] == "heuristic":
    st.caption("⚠️ Models not found — using an Elo heuristic. Run `python -m src.model`.")
else:
    st.caption("Predictions by XGBoost (outcome) + Poisson (scoreline) models.")

# --- Matchday selector -----------------------------------------------------
dates = sorted(fixtures["date"].dt.date.unique())
labels = [d.strftime("%a %d %b %Y") for d in dates]
choice = st.select_slider("Matchday", options=labels, value=labels[0])
day = dates[labels.index(choice)]

day_games = fixtures[fixtures["date"].dt.date == day]
st.subheader(f"{choice}  ·  {len(day_games)} match{'es' if len(day_games) != 1 else ''}")


def _result_color(p: dict) -> str:
    pr = p["probs"]
    top = max(pr, key=pr.get)
    return {"home_win": "🔵", "away_win": "🔴", "draw": "⚪"}[top]


# --- Render each match as a card -------------------------------------------
for g in day_games.itertuples(index=False):
    pred = predict_fixture(g.home_team, g.away_team, bool(g.neutral))
    hg, ag = pred["scoreline"]
    pr = pred["probs"]

    with st.container(border=True):
        venue = f"{g.city}, {g.country}" + ("" if g.neutral else "  🏠")
        st.caption(f"{venue}")
        c1, c2, c3 = st.columns([3, 2, 3])
        c1.markdown(f"### {g.home_team}")
        c2.markdown(f"<h2 style='text-align:center'>{hg} – {ag}</h2>", unsafe_allow_html=True)
        c3.markdown(f"<h3 style='text-align:right'>{g.away_team}</h3>", unsafe_allow_html=True)

        # Probability split as a single stacked bar via three columns.
        st.progress(pr["home_win"], text=f"{g.home_team} win {pr['home_win']*100:.0f}%")
        b1, b2, b3 = st.columns(3)
        b1.metric(f"{g.home_team}", f"{pr['home_win']*100:.0f}%")
        b2.metric("Draw", f"{pr['draw']*100:.0f}%")
        b3.metric(f"{g.away_team}", f"{pr['away_win']*100:.0f}%")
