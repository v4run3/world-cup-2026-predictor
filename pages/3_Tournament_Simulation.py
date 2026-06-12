"""Tournament Simulation page — placeholder for the Monte Carlo bracket (Step 7)."""
import streamlit as st

st.set_page_config(page_title="Tournament Simulation", page_icon="🏆", layout="wide")
st.title("🏆 Tournament Simulation")

st.info(
    "Coming in **Step 7**. This page will Monte-Carlo the full 2026 bracket "
    "thousands of times using the trained models and report each team's "
    "probability of reaching each round and winning the cup.",
    icon="🚧",
)

st.markdown(
    """
**Planned features**
- Group-stage simulation from the official draw
- Knockout bracket resolution using match probabilities
- N-run Monte Carlo → "win the World Cup" % per team
- Most likely final & champion
"""
)
