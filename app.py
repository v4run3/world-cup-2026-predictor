"""World Cup 2026 Predictor — Streamlit entry point (home page).

Run with:  streamlit run app.py
Other pages live in the pages/ folder and appear in the sidebar automatically.
"""
import streamlit as st

import src.data_collection as dc

st.set_page_config(
    page_title="World Cup 2026 Predictor",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ FIFA World Cup 2026 Predictor")
st.caption("Match outcomes, scorelines, win probabilities & team strength — powered by historical data.")

st.markdown(
    """
Welcome! This app predicts the **FIFA World Cup 2026** using machine-learning
models trained on decades of international match data.

**What it does**
- 📅 **Schedule** — every fixture, day by day, with predicted scorelines and win %.
- 📊 **Team Analysis** — explore each nation's strength rating and rank.
- 🏆 **Tournament Simulation** — simulate the whole bracket to a champion.
- 🥅 **Custom Matchup** — predict any hypothetical tie (e.g. a knockout what-if).

Use the sidebar to navigate.
"""
)

# Status banner.
st.success(
    "✅ **Predictions are live**, powered by a trained XGBoost outcome model "
    "(~60% accuracy) and a Poisson scoreline model.",
    icon="✅",
)

with st.expander("Project roadmap"):
    st.markdown(
        """
1. ✅ **Skeleton** — runnable multi-page app
2. ✅ **Data collection** — 49k historical matches + real 2026 fixtures
3. ✅ **Feature engineering** — Elo, form, goal stats (32k training rows)
4. ✅ **Model training** — XGBoost (outcome) + Poisson (scoreline)
5. ✅ **Prediction layer** — models wired into the UI
6. ✅ **Team analysis** — strength breakdowns & charts
7. ⬜ **Tournament simulation** — Monte Carlo the bracket (you are here)
8. ⬜ **Polish & deploy** — Streamlit Community Cloud
"""
    )

teams = st.cache_data(dc.world_cup_2026_teams_safe)()
st.sidebar.success(f"{len(teams)} teams loaded")
