# nfl_spread_simulation_app.py
import streamlit as st
import numpy as np
import pandas as pd

# Try to import nfl_data_py and show a helpful error if it fails
try:
    import nfl_data_py as nfl
except Exception as e:
    st.error(
        "Could not import `nfl_data_py`. "
        "Make sure it's listed in requirements.txt and that you have internet access. "
        f"Import error: {e}"
    )
    st.stop()

# Page config
st.set_page_config(page_title="NFL Spread Monte Carlo", layout="wide")
st.title("🏈 NFL Spread Monte Carlo Simulator")

# Season input
season = st.number_input("Season", min_value=2000, max_value=2025, value=2025, step=1)

@st.cache_data(show_spinner=False)
def load_season_stats(year: int) -> pd.DataFrame:
    """
    Load seasonal team stats using nfl_data_py and compute useful fields.
    This function is cached so repeated interactions are fast.
    """
    stats = nfl.import_seasonal_data([year])
    # Basic required columns check / safe defaults
    if 'games' not in stats.columns:
        # try other names or assume 17
        stats['games'] = stats.get('games_played', 17)

    # points-per-game and opponent ppg
    stats['ppg'] = stats['points'] / stats['games']
    stats['opp_ppg'] = stats['points_against'] / stats['games']

    # net_epa fallback: prefer off_epa/def_epa when available
    if 'off_epa' in stats.columns and 'def_epa' in stats.columns:
        stats['net_epa'] = stats['off_epa'] - stats['def_epa']
    else:
        # fallback: scale point differential per game down so values are similar in magnitude
        stats['net_epa'] = (stats['points'] - stats['points_against']) / stats['games'] / 3.0

    # friendly label for dropdowns
    if 'team_name' in stats.columns:
        stats['label'] = stats['team'] + " - " + stats['team_name']
    else:
        stats['label'] = stats['team']

    # ensure unique and sorted list
    stats = stats.sort_values('label').reset_index(drop=True)
    return stats

# load (or show error)
try:
    stats = load_season_stats(season)
except Exception as e:
    st.error(f"Failed to load season stats for {season}: {e}")
    st.stop()

# UI: team selection + options
col_top_left, col_top_right = st.columns([1, 2])
with col_top_left:
    team_labels = stats['label'].tolist()
    home_label = st.selectbox("Home team", team_labels, index=0)
    away_label = st.selectbox("Away team", [l for l in team_labels if l != home_label], index=0)
    sims = st.slider("Simulations", min_value=1000, max_value=50000, value=10000, step=1000)
    home_field_adv = st.slider("Home field advantage (pts)", 0.0, 4.0, 1.5, step=0.1)

with col_top_right:
    sd_home = st.slider("Home team score SD", 3.0, 12.0, 7.0, step=0.5)
    sd_away = st.slider("Away team score SD", 3.0, 12.0, 6.0, step=0.5)
    vegas_line_input = st.text_input(
        "Optional: sportsbook spread to compare (positive means home favorite). Example: 3.5",
        value=""
    )
    show_sample_margins = st.checkbox("Show sample simulated margins", value=False)

# Run button
if not st.button("Run simulation"):
    st.info("Adjust options then click **Run simulation**.")
    st.stop()

# Parse team codes from label
home_code = home_label.split(" - ")[0]
away_code = away_label.split(" - ")[0]

# Get stats rows
try:
    home_stats = stats.loc[stats['team'] == home_code].iloc[0]
    away_stats = stats.loc[stats['team'] == away_code].iloc[0]
except Exception:
    st.error("Could not find selected teams in the stats dataset. Check the team list and season.")
    st.stop()

# Prepare model inputs
home_ppg = float(home_stats['ppg'])
away_ppg = float(away_stats['ppg'])

# Ratings (use net_epa scaled; fallback handled earlier)
home_rating = float(home_stats['net_epa']) * 100.0
away_rating = float(away_stats['net_epa']) * 100.0

# Calculate model spread (positive = home is favorite by that many points)
model_spread = (home_rating - away_rating) + home_field_adv

# Monte Carlo simulation (use numpy Generator for reproducibility if desired)
rng = np.random.default_rng()
home_scores = rng.normal(loc=home_ppg, scale=sd_home, size=sims)
away_scores = rng.normal(loc=away_ppg, scale=sd_away, size=sims)
margins = home_scores - away_scores  # positive = home wins by margin

# Stats from sims
avg_margin = float(np.mean(margins))
prob_home_cover = float(np.mean(margins > model_spread))
prob_away_cover = float(np.mean(margins < model_spread))

# Display results
st.subheader("📊 Model Results")
col_a, col_b, col_c = st.columns(3)
col_a.metric("Model spread (home)", f"{model_spread:+.1f}")
col_b.metric("Avg simulated margin", f"{avg_margin:.2f}")
col_c.metric(f"Home covers (model)", f"{prob_home_cover*100:.1f}%")

st.write(f"**Away covers (model):** {prob_away_cover*100:.1f}%")
st.write(f"Home team ppg: **{home_ppg:.2f}**, Away team ppg: **{away_ppg:.2f}**")
st.write(f"Home rating: **{home_rating:.2f}**, Away rating: **{away_rating:.2f}**")

# Compare to sportsbook line if provided
if vegas_line_input.strip() != "":
    try:
        vegas_line = float(vegas_line_input)
        prob_home_vs_vegas = float(np.mean(margins > vegas_line))
        st.write(f"**Probability home covers sportsbook line ({vegas_line:+.1f}):** {prob_home_vs_vegas*100:.1f}%")
        # simple "edge" signal
        edge = prob_home_vs_vegas - 0.5
        if edge > 0.03:
            st.success(f"Model suggests a possible positive edge for the home side ({edge*100:.1f} percentage points advantage)")
        elif edge < -0.03:
            st.info(f"Model suggests market favors the home side more than model (edge {edge*100:.1f} pp)")
        else:
            st.write(f"No strong edge vs market (edge {edge*100:.1f} percentage points)")
    except ValueError:
        st.warning("Couldn't parse the sportsbook line. Enter a number like 3.5 or -2")

# Histogram of simulated margins
st.subheader("Distribution of simulated margins (home - away)")
df_m = pd.DataFrame({"margin": margins})
# use altair (streamlit supports it) for a nicely binned histogram
try:
    import altair as alt
    chart = (
        alt.Chart(df_m)
        .mark_bar()
        .encode(
            alt.X("margin:Q", bin=alt.Bin(maxbins=50), title="Home - Away margin"),
            alt.Y("count()", title="Simulations"),
            tooltip=[alt.Tooltip("count()", title="Count")]
        )
        .properties(width=800, height=320)
    )
    st.altair_chart(chart, use_container_width=True)
except Exception:
    # fallback: show simple pandas histogram
    st.write(df_m['margin'].describe())
    st.bar_chart(pd.cut(df_m['margin'], bins=40).value_counts().sort_index())

# Optional sample margins
if show_sample_margins:
    st.write("First 50 simulated margins (home - away):")
    st.write(pd.Series(margins).round(1).head(50).to_list())

st.caption("Model = net EPA (or ppg-based fallback) scaled → Monte Carlo simulation. "
           "This is a simplified model for teaching and experimentation.")
