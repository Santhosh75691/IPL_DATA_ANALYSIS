# ============================================================
# IPL DATA ANALYSIS
# Beginner Data Analytics Project
# ============================================================

# Install libraries if required:
# pip install pandas numpy matplotlib seaborn openpyxl

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# STEP 1: Load the datasets
# ------------------------------------------------------------

matches = pd.read_csv("ipl_match_data(1).csv")
performance = pd.read_csv("ipl_player_performance(1).csv")

matches["Date"] = pd.to_datetime(matches["Date"])
performance["Date"] = pd.to_datetime(performance["Date"])

print("\n========== MATCH DATA ==========")
print(matches.head())

print("\n========== PLAYER PERFORMANCE DATA ==========")
print(performance.head())

print("\nMatch rows:", len(matches))
print("Performance rows:", len(performance))


# ------------------------------------------------------------
# STEP 2: Understand the data
# ------------------------------------------------------------

print("\n========== MATCH DATA TYPES ==========")
print(matches.dtypes)

print("\n========== PERFORMANCE DATA TYPES ==========")
print(performance.dtypes)

print("\n========== MATCH MISSING VALUES ==========")
print(matches.isnull().sum())

print("\n========== PERFORMANCE MISSING VALUES ==========")
print(performance.isnull().sum())

print("\n========== MATCH DUPLICATES ==========")
print(matches.duplicated().sum())

print("\n========== PERFORMANCE DUPLICATES ==========")
print(performance.duplicated().sum())


# ------------------------------------------------------------
# STEP 3: Clean the data
# ------------------------------------------------------------

matches = matches.drop_duplicates()
performance = performance.drop_duplicates()

# Fill categorical missing values
for col in ["City", "Venue", "Toss_Winner", "Toss_Decision",
            "Winner", "Player_of_Match"]:
    if matches[col].isnull().sum() > 0:
        matches[col] = matches[col].fillna(matches[col].mode()[0])

# Fill numeric missing values
numeric_cols = [
    "Runs", "Balls_Faced", "Fours", "Sixes", "Wickets",
    "Overs_Bowled", "Runs_Conceded", "Strike_Rate", "Economy_Rate"
]

for col in numeric_cols:
    if performance[col].isnull().sum() > 0:
        performance[col] = performance[col].fillna(0)


# ------------------------------------------------------------
# STEP 4: Feature Engineering
# ------------------------------------------------------------

matches["Year"] = matches["Date"].dt.year
matches["Month"] = matches["Date"].dt.month
matches["Month_Name"] = matches["Date"].dt.strftime("%b")

matches["Season"] = matches["Date"].dt.year
performance["Season"] = performance["Date"].dt.year

# Did the toss winner also win the match?
matches["Toss_Winner_Match_Winner"] = np.where(
    matches["Toss_Winner"] == matches["Winner"],
    "Yes",
    "No"
)


# ------------------------------------------------------------
# STEP 5: Overall KPIs
# ------------------------------------------------------------

total_matches = matches["Match_ID"].nunique()

total_runs = performance["Runs"].sum()

total_wickets = performance["Wickets"].sum()

average_runs_per_player_record = performance["Runs"].mean()

total_sixes = performance["Sixes"].sum()

total_fours = performance["Fours"].sum()

toss_win_match_win_pct = (
    matches["Toss_Winner_Match_Winner"].eq("Yes").mean() * 100
)

print("\n========== IPL KPIs ==========")

print(f"Total Matches              : {total_matches:,}")
print(f"Total Runs                 : {total_runs:,}")
print(f"Total Wickets              : {total_wickets:,}")
print(f"Total Sixes                : {total_sixes:,}")
print(f"Total Fours                : {total_fours:,}")
print(f"Average Runs               : {average_runs_per_player_record:.2f}")
print(f"Toss Winner Also Won Match: {toss_win_match_win_pct:.2f}%")


# ------------------------------------------------------------
# STEP 6: Team Performance
# ------------------------------------------------------------

team_list = sorted(set(matches["Team1"]) | set(matches["Team2"]))

team_rows = []

for team in team_list:
    matches_played = (
        (matches["Team1"] == team) |
        (matches["Team2"] == team)
    ).sum()

    wins = (matches["Winner"] == team).sum()

    losses = matches_played - wins

    win_percentage = (wins / matches_played) * 100 if matches_played else 0

    team_rows.append([
        team,
        matches_played,
        wins,
        losses,
        round(win_percentage, 2)
    ])

team_analysis = pd.DataFrame(
    team_rows,
    columns=[
        "Team",
        "Matches_Played",
        "Wins",
        "Losses",
        "Win_Percentage"
    ]
)

team_analysis = team_analysis.sort_values(
    "Wins",
    ascending=False
)

print("\n========== TEAM PERFORMANCE ==========")
print(team_analysis)


# ------------------------------------------------------------
# STEP 7: Player Batting Analysis
# ------------------------------------------------------------

player_batting = (
    performance
    .groupby("Player", as_index=False)
    .agg(
        Runs=("Runs", "sum"),
        Balls_Faced=("Balls_Faced", "sum"),
        Fours=("Fours", "sum"),
        Sixes=("Sixes", "sum")
    )
)

player_batting["Strike_Rate"] = (
    player_batting["Runs"] /
    player_batting["Balls_Faced"] * 100
)

player_batting = player_batting.sort_values(
    "Runs",
    ascending=False
)

print("\n========== TOP BATTERS ==========")
print(player_batting.head(10))


# ------------------------------------------------------------
# STEP 8: Player Bowling Analysis
# ------------------------------------------------------------

player_bowling = (
    performance
    .groupby("Player", as_index=False)
    .agg(
        Wickets=("Wickets", "sum"),
        Overs=("Overs_Bowled", "sum"),
        Runs_Conceded=("Runs_Conceded", "sum")
    )
)

player_bowling["Economy_Rate"] = (
    player_bowling["Runs_Conceded"] /
    player_bowling["Overs"]
)

player_bowling["Economy_Rate"] = (
    player_bowling["Economy_Rate"].replace(
        [np.inf, -np.inf],
        0
    )
)

player_bowling = player_bowling.sort_values(
    "Wickets",
    ascending=False
)

print("\n========== TOP BOWLERS ==========")
print(player_bowling.head(10))


# ------------------------------------------------------------
# STEP 9: Player of the Match Analysis
# ------------------------------------------------------------

pom_analysis = (
    matches["Player_of_Match"]
    .value_counts()
    .reset_index()
)

pom_analysis.columns = [
    "Player",
    "Player_of_Match_Awards"
]

print("\n========== PLAYER OF THE MATCH ==========")
print(pom_analysis.head(10))


# ------------------------------------------------------------
# STEP 10: Toss Analysis
# ------------------------------------------------------------

toss_analysis = (
    matches["Toss_Decision"]
    .value_counts()
    .reset_index()
)

toss_analysis.columns = [
    "Toss_Decision",
    "Matches"
]

print("\n========== TOSS DECISION ==========")
print(toss_analysis)

toss_result_analysis = (
    matches["Toss_Winner_Match_Winner"]
    .value_counts()
    .reset_index()
)

toss_result_analysis.columns = [
    "Toss_Winner_Match_Winner",
    "Matches"
]

print("\n========== TOSS WINNER VS MATCH WINNER ==========")
print(toss_result_analysis)


# ------------------------------------------------------------
# STEP 11: Venue Analysis
# ------------------------------------------------------------

venue_analysis = (
    matches
    .groupby("Venue", as_index=False)
    .agg(
        Matches=("Match_ID", "nunique")
    )
    .sort_values(
        "Matches",
        ascending=False
    )
)

print("\n========== VENUE ANALYSIS ==========")
print(venue_analysis)


# ------------------------------------------------------------
# STEP 12: Season Analysis
# ------------------------------------------------------------

season_analysis = (
    matches
    .groupby("Season", as_index=False)
    .agg(
        Matches=("Match_ID", "nunique")
    )
)

season_runs = (
    performance
    .groupby("Season", as_index=False)
    .agg(
        Runs=("Runs", "sum"),
        Wickets=("Wickets", "sum")
    )
)

season_analysis = season_analysis.merge(
    season_runs,
    on="Season"
)

print("\n========== SEASON ANALYSIS ==========")
print(season_analysis)


# ------------------------------------------------------------
# STEP 13: Create Charts
# ------------------------------------------------------------

sns.set_theme(style="whitegrid")


# Chart 1: Team Wins
plt.figure(figsize=(12, 6))

sns.barplot(
    data=team_analysis,
    x="Team",
    y="Wins"
)

plt.title("IPL Team Wins")

plt.xlabel("Team")

plt.ylabel("Wins")

plt.xticks(rotation=45, ha="right")

plt.tight_layout()

plt.show()


# Chart 2: Team Win Percentage
plt.figure(figsize=(12, 6))

sns.barplot(
    data=team_analysis.sort_values(
        "Win_Percentage",
        ascending=False
    ),
    x="Team",
    y="Win_Percentage"
)

plt.title("Team Win Percentage")

plt.xlabel("Team")

plt.ylabel("Win Percentage (%)")

plt.xticks(rotation=45, ha="right")

plt.tight_layout()

plt.show()


# Chart 3: Top 10 Run Scorers
top_run_scorers = player_batting.head(10)

plt.figure(figsize=(12, 6))

sns.barplot(
    data=top_run_scorers,
    y="Player",
    x="Runs"
)

plt.title("Top 10 Run Scorers")

plt.xlabel("Total Runs")

plt.ylabel("Player")

plt.tight_layout()

plt.show()


# Chart 4: Top 10 Wicket Takers
top_wicket_takers = player_bowling.head(10)

plt.figure(figsize=(12, 6))

sns.barplot(
    data=top_wicket_takers,
    y="Player",
    x="Wickets"
)

plt.title("Top 10 Wicket Takers")

plt.xlabel("Total Wickets")

plt.ylabel("Player")

plt.tight_layout()

plt.show()


# Chart 5: Sixes by Top Batters
top_six_hitters = (
    player_batting
    .sort_values("Sixes", ascending=False)
    .head(10)
)

plt.figure(figsize=(12, 6))

sns.barplot(
    data=top_six_hitters,
    y="Player",
    x="Sixes"
)

plt.title("Top 10 Six Hitters")

plt.xlabel("Sixes")

plt.ylabel("Player")

plt.tight_layout()

plt.show()


# Chart 6: Player of Match Awards
top_pom = pom_analysis.head(10)

plt.figure(figsize=(12, 6))

sns.barplot(
    data=top_pom,
    y="Player",
    x="Player_of_Match_Awards"
)

plt.title("Top Player of the Match Award Winners")

plt.xlabel("Awards")

plt.ylabel("Player")

plt.tight_layout()

plt.show()


# Chart 7: Toss Decision
plt.figure(figsize=(8, 6))

sns.barplot(
    data=toss_analysis,
    x="Toss_Decision",
    y="Matches"
)

plt.title("Toss Decision Analysis")

plt.xlabel("Toss Decision")

plt.ylabel("Number of Matches")

plt.tight_layout()

plt.show()


# Chart 8: Toss Winner vs Match Winner
plt.figure(figsize=(8, 6))

sns.barplot(
    data=toss_result_analysis,
    x="Toss_Winner_Match_Winner",
    y="Matches"
)

plt.title("Toss Winner vs Match Winner")

plt.xlabel("Did Toss Winner Win Match?")

plt.ylabel("Number of Matches")

plt.tight_layout()

plt.show()


# Chart 9: Matches by Season
plt.figure(figsize=(10, 6))

sns.barplot(
    data=season_analysis,
    x="Season",
    y="Matches"
)

plt.title("Matches by IPL Season")

plt.xlabel("Season")

plt.ylabel("Matches")

plt.tight_layout()

plt.show()


# Chart 10: Total Runs by Season
plt.figure(figsize=(10, 6))

sns.lineplot(
    data=season_analysis,
    x="Season",
    y="Runs",
    marker="o"
)

plt.title("Total Runs by IPL Season")

plt.xlabel("Season")

plt.ylabel("Runs")

plt.tight_layout()

plt.show()


# Chart 11: Venue Matches
top_venues = venue_analysis.head(10)

plt.figure(figsize=(12, 6))

sns.barplot(
    data=top_venues,
    y="Venue",
    x="Matches"
)

plt.title("Top Venues by Number of Matches")

plt.xlabel("Matches")

plt.ylabel("Venue")

plt.tight_layout()

plt.show()


# Chart 12: Batting Strike Rate vs Runs
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=player_batting,
    x="Runs",
    y="Strike_Rate",
    s=100
)

plt.title("Runs vs Strike Rate")

plt.xlabel("Total Runs")

plt.ylabel("Strike Rate")

plt.tight_layout()

plt.show()


# Chart 13: Correlation Heatmap
correlation_cols = [
    "Runs",
    "Balls_Faced",
    "Fours",
    "Sixes",
    "Wickets",
    "Overs_Bowled",
    "Runs_Conceded",
    "Strike_Rate",
    "Economy_Rate"
]

plt.figure(figsize=(11, 8))

sns.heatmap(
    performance[correlation_cols].corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("IPL Player Performance Correlation")

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# STEP 14: Export Analysis Results
# ------------------------------------------------------------

matches.to_csv(
    "ipl_match_data_cleaned.csv",
    index=False
)

performance.to_csv(
    "ipl_player_performance_cleaned.csv",
    index=False
)


# Export all analysis tables to Excel

with pd.ExcelWriter(
    "ipl_data_analysis_report.xlsx",
    engine="openpyxl"
) as writer:

    matches.to_excel(
        writer,
        sheet_name="Match_Data",
        index=False
    )

    performance.to_excel(
        writer,
        sheet_name="Player_Performance",
        index=False
    )

    team_analysis.to_excel(
        writer,
        sheet_name="Team_Analysis",
        index=False
    )

    player_batting.to_excel(
        writer,
        sheet_name="Batting_Analysis",
        index=False
    )

    player_bowling.to_excel(
        writer,
        sheet_name="Bowling_Analysis",
        index=False
    )

    pom_analysis.to_excel(
        writer,
        sheet_name="POM_Analysis",
        index=False
    )

    toss_analysis.to_excel(
        writer,
        sheet_name="Toss_Analysis",
        index=False
    )

    venue_analysis.to_excel(
        writer,
        sheet_name="Venue_Analysis",
        index=False
    )

    season_analysis.to_excel(
        writer,
        sheet_name="Season_Analysis",
        index=False
    )


print("\n================================================")
print("IPL DATA ANALYSIS PROJECT COMPLETED")
print("================================================")

print("\nFiles created:")
print("1. ipl_match_data_cleaned.csv")
print("2. ipl_player_performance_cleaned.csv")
print("3. ipl_data_analysis_report.xlsx")

print("\nYou can now import the cleaned data into Power BI.")
