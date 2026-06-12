"""Custom Matchup page — pick any two teams (e.g. a hypothetical knockout tie)."""
import plotly.graph_objects as go
import streamlit as st

import src.data_collection as dc
from src.predict import predict_match

st.set_page_config(page_title="Custom Matchup", page_icon="🥅", layout="wide")
st.title("🥅 Custom Matchup")
st.caption("Predict any hypothetical fixture — handy for knockout 'what-if' ties.")

teams = st.cache_data(dc.world_cup_2026_teams_safe)()
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    home = st.selectbox("Team A", teams, index=0)
with col2:
    away = st.selectbox("Team B", teams, index=1)
with col3:
    neutral = st.toggle("Neutral venue", value=True)

if home == away:
    st.warning("Pick two different teams.")
    st.stop()

if st.button("Predict", type="primary", use_container_width=True):
    result = predict_match(home, away, neutral=neutral)
    probs = result["probs"]

    if result["source"] == "heuristic":
        st.caption("⚠️ Models not found — using an Elo heuristic. Run `python -m src.model`.")

    # --- Probabilities -----------------------------------------------------
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{home} win", f"{probs['home_win'] * 100:.0f}%")
    c2.metric("Draw", f"{probs['draw'] * 100:.0f}%")
    c3.metric(f"{away} win", f"{probs['away_win'] * 100:.0f}%")

    # Probability bar
    fig = go.Figure(go.Bar(
        x=[probs["home_win"], probs["draw"], probs["away_win"]],
        y=[f"{home} win", "Draw", f"{away} win"],
        orientation="h",
        marker_color=["#2563eb", "#9ca3af", "#dc2626"],
    ))
    fig.update_layout(xaxis_tickformat=".0%", height=220, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # --- Scoreline & strength ---------------------------------------------
    hg, ag = result["scoreline"]
    st.subheader(f"Predicted scoreline:  {home} {hg} – {ag} {away}")

    s = result["strength"]
    sc1, sc2 = st.columns(2)
    sc1.metric(f"{home} strength", f"{s['home']}/100")
    sc2.metric(f"{away} strength", f"{s['away']}/100")
