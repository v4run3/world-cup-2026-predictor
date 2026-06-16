"""Schedule page — full FIFA World Cup 2026 fixture list, Jun 11 → Jul 19.

Group stage matches show actual scores (played) or model predictions (upcoming).
Knockout matches show TBD until teams are confirmed after the group stage.
"""

import datetime

import pandas as pd
import streamlit as st

import src.data_collection as dc
from src.predict import predict_match

st.set_page_config(page_title="Schedule", page_icon="📅", layout="wide")
st.title("📅 Match Schedule & Predictions")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@st.cache_data
def get_full_schedule() -> pd.DataFrame:
    """Merge group-stage fixtures with knockout placeholders into one table."""
    group = dc.load_all_wc2026_matches()
    group["match_round"] = "Group Stage"

    ko = dc.load_knockout_placeholders()

    # Align dtypes before concat
    for col in ["home_score", "away_score"]:
        group[col] = group[col].astype("Int64")

    all_matches = pd.concat([group, ko], ignore_index=True)
    return all_matches.sort_values("date").reset_index(drop=True)


@st.cache_data
def predict_fixture(home: str, away: str, neutral: bool) -> dict:
    return predict_match(home, away, neutral=neutral)


fixtures = get_full_schedule()

if fixtures.empty:
    st.warning("No fixtures found. Run `python -m src.data_collection --download`.")
    st.stop()

# Model-source banner
gs_rows = fixtures[fixtures["home_team"] != "TBD"]
if not gs_rows.empty:
    first = gs_rows.iloc[0]
    sample = predict_fixture(
        str(first.home_team), str(first.away_team), bool(first.neutral)
    )
    if sample["source"] == "heuristic":
        st.caption(
            "⚠️ Models not found — using an Elo heuristic. Run `python -m src.model`."
        )
    else:
        st.caption(
            "✅ Played matches show the **actual final score**  ·  "
            "🔮 Upcoming group matches show **model predictions**  ·  "
            "⏳ Knockout slots show **TBD** until teams qualify."
        )

# ---------------------------------------------------------------------------
# Matchday / round selector — selectbox grouped by stage
# ---------------------------------------------------------------------------

# Build one option per date, labelled by round and formatted date.
STAGE_EMOJI = {
    "Group Stage": "⚽",
    "Round of 32": "🔵",
    "Round of 16": "⚡",
    "Quarter-final": "🏅",
    "Semi-final": "🔥",
    "3rd Place": "🥉",
    "Final": "🏆",
}

dates_in_order: list[datetime.date] = []
date_to_label: dict[datetime.date, str] = {}

for d_ts, grp in fixtures.groupby(fixtures["date"].dt.date):
    d = d_ts  # already a datetime.date from groupby
    stage = grp["match_round"].iloc[0]
    emoji = STAGE_EMOJI.get(stage, "📅")
    n = len(grp)
    label = f"{emoji}  {d.strftime('%a %d %b')}  —  {stage}  ({n} match{'es' if n > 1 else ''})"
    dates_in_order.append(d)
    date_to_label[d] = label

# Default to today if it falls inside the tournament, otherwise first matchday.
today = datetime.date.today()
default_idx = 0
for i, d in enumerate(dates_in_order):
    if d >= today:
        default_idx = i
        break

all_labels = [date_to_label[d] for d in dates_in_order]

chosen_label = st.selectbox(
    "Select matchday",
    options=all_labels,
    index=default_idx,
    label_visibility="collapsed",
)
chosen_day = dates_in_order[all_labels.index(chosen_label)]
day_games = fixtures[fixtures["date"].dt.date == chosen_day]

# Summary header
stage = day_games["match_round"].iloc[0]
n_played = int(day_games["is_played"].sum())
n_tbd = int((day_games["home_team"] == "TBD").sum())
n_upcoming = len(day_games) - n_played - n_tbd

parts = []
if n_played:
    parts.append(f"✅ {n_played} played")
if n_upcoming:
    parts.append(f"🔮 {n_upcoming} upcoming")
if n_tbd:
    parts.append(f"⏳ {n_tbd} TBD")

st.subheader(
    f"{STAGE_EMOJI.get(stage, '📅')} {stage}  ·  "
    f"{chosen_day.strftime('%A, %d %B %Y')}  ·  "
    f"{len(day_games)} match{'es' if len(day_games) != 1 else ''}"
    + (f"  ({', '.join(parts)})" if parts else "")
)

# ---------------------------------------------------------------------------
# Render each match card
# ---------------------------------------------------------------------------

for g in day_games.itertuples(index=False):
    is_tbd = str(g.home_team) == "TBD"
    venue = f"{g.city}, {g.country}"

    with st.container(border=True):
        if is_tbd:
            # ── Knockout placeholder ────────────────────────────────────────
            st.caption(f"⏳ **{g.match_round}** · {venue}")
            c1, c2, c3 = st.columns([3, 2, 3])
            c1.markdown("### TBD")
            c2.markdown(
                "<h2 style='text-align:center;color:#aaa'>vs</h2>",
                unsafe_allow_html=True,
            )
            c3.markdown(
                "<h3 style='text-align:right'>TBD</h3>",
                unsafe_allow_html=True,
            )
            st.caption("Teams will be known once the group stage is complete.")

        elif g.is_played:
            # ── Played group-stage match ────────────────────────────────────
            pred = predict_fixture(str(g.home_team), str(g.away_team), bool(g.neutral))
            pred_h, pred_a = pred["scoreline"]
            actual_h, actual_a = int(g.home_score), int(g.away_score)

            if actual_h > actual_a:
                result_label = f"🔵 {g.home_team} win"
            elif actual_a > actual_h:
                result_label = f"🔴 {g.away_team} win"
            else:
                result_label = "⚪ Draw"

            st.caption(f"✅ **FULL TIME** · {venue}")
            c1, c2, c3 = st.columns([3, 2, 3])
            c1.markdown(f"### {g.home_team}")
            c2.markdown(
                f"<h2 style='text-align:center'>{actual_h} – {actual_a}</h2>",
                unsafe_allow_html=True,
            )
            c3.markdown(
                f"<h3 style='text-align:right'>{g.away_team}</h3>",
                unsafe_allow_html=True,
            )
            pred_icon = "🎯" if (pred_h == actual_h and pred_a == actual_a) else "🔮"
            st.caption(
                f"{pred_icon} Model predicted: **{g.home_team} {pred_h}–{pred_a} {g.away_team}**"
                f"  ·  {result_label}"
            )

        else:
            # ── Upcoming group-stage match ──────────────────────────────────
            pred = predict_fixture(str(g.home_team), str(g.away_team), bool(g.neutral))
            pred_h, pred_a = pred["scoreline"]
            pr = pred["probs"]

            st.caption(f"🔮 **UPCOMING** · {venue}")
            c1, c2, c3 = st.columns([3, 2, 3])
            c1.markdown(f"### {g.home_team}")
            c2.markdown(
                f"<h2 style='text-align:center'>{pred_h} – {pred_a}</h2>",
                unsafe_allow_html=True,
            )
            c3.markdown(
                f"<h3 style='text-align:right'>{g.away_team}</h3>",
                unsafe_allow_html=True,
            )
            st.progress(
                pr["home_win"],
                text=f"{g.home_team} win  {pr['home_win'] * 100:.0f}%",
            )
            b1, b2, b3 = st.columns(3)
            b1.metric(f"{g.home_team}", f"{pr['home_win'] * 100:.0f}%")
            b2.metric("Draw", f"{pr['draw'] * 100:.0f}%")
            b3.metric(f"{g.away_team}", f"{pr['away_win'] * 100:.0f}%")
