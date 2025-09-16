import numpy as np
import nfl_data_py as nfl
import pandas as pd

# --- NFL Spread Monte Carlo Simulation with Live Data & Team Selection ---

# Choose year
season = 2025

# Load season data
stats = nfl.import_seasonal_data([season])
stats['ppg'] = stats['points'] / stats['games']
stats['opp_ppg'] = stats['points_against'] / stats['games']
stats['net_epa'] = stats['off_epa'] - stats['def_epa']

# Display available team codes
print("Available teams:", stats['team'].unique())

# User input
team_a = input("Enter home team code (e.g., KAN for Chiefs): ").strip().upper()
team_b = input("Enter away team code (e.g., LVR for Raiders): ").strip().upper()

# Get stats
team_a_stats = stats.loc[stats['team']==team_a].iloc[0]
team_b_stats = stats.loc[stats['team']==team_b].iloc[0]

# Power ratings (using net EPA scaled)
team_a_rating = team_a_stats['net_epa'] * 100
team_b_rating = team_b_stats['net_epa'] * 100

# Home field advantage for Team A
spread = (team_a_rating - team_b_rating) + 1.5

# Use PPG as scoring baseline
team_a_ppg = team_a_stats['ppg']
team_b_ppg = team_b_stats['ppg']

# Standard deviation (can refine with real variance)
team_a_sd = 7
team_b_sd = 6

# Monte Carlo Simulation
sims = 10000
team_a_scores = np.random.normal(team_a_ppg, team_a_sd, sims)
team_b_scores = np.random.normal(team_b_ppg, team_b_sd, sims)

margins = team_a_scores - team_b_scores

# Results
avg_margin = np.mean(margins)
cover_prob = np.mean(margins > spread)  # Home team covers
away_cover_prob = np.mean(margins < spread)  # Away team covers

print(f"Projected Spread: {team_a} {spread:+.1f}")
print(f"Average Simulated Margin: {avg_margin:.2f}")
print(f"Probability {team_a} Covers: {cover_prob*100:.1f}%")
print(f"Probability {team_b} Covers: {away_cover_prob*100:.1f}%")
