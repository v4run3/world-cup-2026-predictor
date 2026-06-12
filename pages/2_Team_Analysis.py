"""Team Analysis page — real Elo strength rating per nation."""
import streamlit as st

import src.data_collection as dc
from src.features import load_elo_ratings
from src.predict import _to_strength_score

st.set_page_config(page_title="Team Analysis", page_icon="📊", layout="wide")
st.title("📊 Team Analysis")

teams = st.cache_data(dc.world_cup_2026_teams_safe)()
elo_map = st.cache_data(load_elo_ratings)()

if not elo_map:
    st.warning("Elo ratings not built yet — run `python -m src.features`.")
    st.stop()
st.caption("Strength = Elo rating computed over the full 1872–2026 match history.")


def get_elo(t: str) -> float:
    return float(elo_map.get(t, 1500))


team = st.selectbox("Select a team", teams)
elo = get_elo(team)

# Rank among the 48 qualified teams.
rank = sorted(teams, key=get_elo, reverse=True).index(team) + 1

c1, c2, c3 = st.columns(3)
c1.metric("Strength score", f"{_to_strength_score(elo)}/100")
c2.metric("Elo rating", f"{elo:.0f}")
c3.metric("Rank (of 48)", f"#{rank}")

st.divider()
st.subheader("All qualified teams by strength")

ranked = sorted(teams, key=get_elo, reverse=True)
chart_data = {t: _to_strength_score(get_elo(t)) for t in ranked}
st.bar_chart(chart_data, horizontal=True)
