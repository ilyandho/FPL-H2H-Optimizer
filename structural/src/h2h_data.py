"""Inputs of the 2024-25 head-to-head stage (h2h.py, irl_h2h.py), built once from their sources.

  data/24-25/league/player_features.csv   every player in the FPL pool each round, with the 14 features the league
                                          selection model (league betas) was fitted on, price, position and outcome:
                                          the columns of 24-25/optimizer/inputs/rolled_data_24-25_xp_fixed.csv that
                                          h2h_opponents.py uses (its pools must match league_selections' binary columns)
  data/24-25/league/league_selections.csv the league's squads per round, with the per-position ownership vectors
                                          (24-25/optimizer/inputs/league_selections_df.csv, unchanged)
  data/24-25/ffs/picks.csv.gz             Fantasy Football Scout H2H league managers' picks per gameweek: 15 players with
                                          their slot (1-11 starting, 12-15 bench in auto-sub order), multiplier, captain
                                          and vice (data/sources/ffs_24-25/FFS_manager_weekly_selections*.csv, from
                                          FPL-Optimal-Transfer's "FPL Data"); the Assistant Manager chip's 16th row
                                          (element_type 5, not a player) is dropped
  data/24-25/ffs/managers.csv             every FFS manager's league (1 = top of the promotion/relegation pyramid) and
                                          division, and FPL overall rank at the scrape (FFS_League_{1..10}.csv); names
                                          are left out

The league selection model's fitted betas and manager fits stay where 24-25/optimizer/notebooks wrote them
(config.SEASONS["24-25"]["league_models"]); they are about 150 MB.

Usage (from src/): python h2h_data.py
"""

import os

import pandas as pd

import config

ROLLED = "../../opponent_model/inputs/rolled_data_24-25_xp_fixed.csv.gz"
LEAGUE_SELECTIONS = "../../opponent_model/inputs/league_selections_df.csv"
FFS_SOURCE = f"{config.SOURCES}/ffs_24-25"
FEATURES = ["assists_3_ewm", "blocks_3_ewm", "clearances_3_ewm", "difficulty", "elo_diff", "goals_conceded_3_ewm",
            "goals_prevented_3_ewm", "interceptions_3_ewm", "opponent_difficulty", "percentage_net_transfers",
            "points_3_ewm", "points_5_ewm", "selected_by_percent", "yellow_cards_rolling_5"]
KEEP = ["round", "element", "position", "name", "team", "opponent_team", "value", "minutes", "total_points", "xP"]


def main():
    p = config.paths("24-25")
    os.makedirs(p.league_dir, exist_ok=True)
    os.makedirs(p.ffs_dir, exist_ok=True)

    rolled = pd.read_csv(ROLLED, usecols=KEEP + FEATURES, low_memory=False)
    rolled.to_csv(f"{p.league_dir}/player_features.csv", index=False)
    pd.read_csv(LEAGUE_SELECTIONS).to_csv(f"{p.league_dir}/league_selections.csv", index=False)
    print(f"league: {len(rolled)} player-rounds, {rolled['round'].nunique()} rounds")

    picks = pd.concat([pd.read_csv(f"{FFS_SOURCE}/{f}") for f in
                       ["FFS_manager_weekly_selections.csv.gz", "FFS_manager_weekly_selections2.csv"]])
    picks = picks[picks["element_type"] != 5].drop_duplicates(["team_id", "game_week", "element"])
    picks = picks.rename(columns={"game_week": "gw", "position": "slot"})[
        ["team_id", "gw", "element", "slot", "multiplier", "is_captain", "is_vice_captain"]]
    picks.sort_values(["team_id", "gw", "slot"]).to_csv(f"{p.ffs_dir}/picks.csv.gz", index=False)

    managers = pd.concat([pd.read_csv(f"{FFS_SOURCE}/FFS_League_{i}.csv") for i in range(1, 11)])
    managers = managers.rename(columns={"Team id": "team_id", "League": "league", "Division": "division",
                                        "OR": "overall_rank"})
    managers["overall_rank"] = pd.to_numeric(managers["overall_rank"].astype(str).str.replace(",", ""),
                                             errors="coerce")
    managers = managers.drop_duplicates("team_id")[["team_id", "league", "division", "overall_rank"]]
    managers["has_picks"] = managers["team_id"].isin(picks["team_id"])
    managers.to_csv(f"{p.ffs_dir}/managers.csv", index=False)
    sizes = picks.groupby(["team_id", "gw"]).size()
    print(f"ffs: {picks['team_id'].nunique()} managers with picks, {len(sizes)} manager-gameweeks "
          f"({(sizes == 15).mean():.1%} with 15 players); managers by league with picks: "
          f"{managers[managers['has_picks']].groupby('league').size().to_dict()}")


if __name__ == "__main__":
    main()
