"""Tournament Simulation page — Monte-Carlo the whole World Cup to a champion."""
import plotly.express as px
import streamlit as st

from src.simulate import TournamentSimulator

st.set_page_config(page_title="Tournament Simulation", page_icon="🏆", layout="wide")
st.title("🏆 Tournament Simulation")
st.caption(
    "Simulates the full 48-team tournament many times using the Poisson goal "
    "model — group stage, qualification, and a seeded knockout bracket — to "
    "estimate each team's title chances."
)


@st.cache_resource
def get_simulator() -> TournamentSimulator:
    return TournamentSimulator()


@st.cache_data
def run_sim(n_sims: int):
    return get_simulator().run(n_sims=n_sims)


col1, col2 = st.columns([3, 1])
n_sims = col1.select_slider(
    "Number of simulations",
    options=[500, 1000, 2000, 5000, 10000],
    value=2000,
    help="More simulations = smoother estimates but slower.",
)
go = col2.button("▶ Run simulation", type="primary", use_container_width=True)

if go or "sim_table" not in st.session_state:
    with st.spinner(f"Running {n_sims:,} tournaments..."):
        st.session_state.sim_table = run_sim(n_sims)
        st.session_state.sim_n = n_sims

table = st.session_state.sim_table
st.caption(f"Based on {st.session_state.sim_n:,} simulated tournaments.")

# --- Champion odds ---------------------------------------------------------
st.subheader("🥇 Title odds")
top = table.head(15).iloc[::-1]  # reverse so highest is on top
fig = px.bar(
    top, x="Win Cup %", y="team", orientation="h",
    text="Win Cup %", color="Win Cup %", color_continuous_scale="Blues",
)
fig.update_layout(height=500, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
fig.update_traces(texttemplate="%{text}%", textposition="outside")
st.plotly_chart(fig, use_container_width=True)

# --- Full progression table ------------------------------------------------
st.subheader("📋 Round-by-round probabilities")
group_filter = st.multiselect(
    "Filter by group", sorted(table["group"].unique()), default=[]
)
shown = table[table["group"].isin(group_filter)] if group_filter else table
st.dataframe(
    shown,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Win Cup %": st.column_config.ProgressColumn(
            "Win Cup %", min_value=0, max_value=float(table["Win Cup %"].max()), format="%.1f%%"
        ),
    },
)
