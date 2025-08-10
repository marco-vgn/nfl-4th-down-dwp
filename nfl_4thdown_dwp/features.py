"""
Extracts 4th downs rows for labels calculation.
Saves to data/interim/fourthdown_states.parquet
Usage:
  python -m nfl_4thdown_dwp.features
"""
import os
import pandas as pd

RAW_GLOB = "data/raw/pbp_*.parquet"
OUT_PATH = "data/interim/fourthdown_states.csv"

KEEP_COLS = [
    # no-team state
    "season","game_id","play_id",
    "qtr","game_seconds_remaining","half_seconds_remaining",
    "yardline_100","ydstogo","score_differential",
    "posteam","defteam","home_team","away_team","week",
    "posteam_timeouts_remaining","defteam_timeouts_remaining",
    "roof","surface","temp","wind","spread_line",
    # play state
    "down","play_type","season_type"
]


def normalize_cats(df: pd.DataFrame) -> pd.DataFrame:
    # fill na for roof and surface
    df["roof"] = df["roof"].fillna("outdoors").str.lower()
    df["surface"] = df["surface"].fillna("grass").str.lower()
    # rename columns with measure units
    df = df.rename(columns={"temp":"temp_f","wind":"wind_mph","qtr":"quarter"})
    # timeouts
    df = df.rename(columns={
        "posteam_timeouts_remaining":"timeouts_off",
        "defteam_timeouts_remaining":"timeouts_def"
    })
    # Set receiving 2nd half kick off booleans
    df["receive_2h_ko"] = df["receive_2h_ko"].fillna(False).astype(bool)
    return df

def main():
    os.makedirs("data/interim", exist_ok=True)
    # get parquet files in data/raw
    import glob
    files = sorted(glob.glob(RAW_GLOB))
    if not files:
        raise SystemExit("Data not found in data/raw/. Run dataset.py first.")
    dfs = [pd.read_parquet(p, columns=KEEP_COLS) for p in files]
    pbp = pd.concat(dfs, ignore_index=True)

    # derive second-half receiver and receive_2h_ko
    pbp_sorted = pbp.sort_values(["game_id","qtr","game_seconds_remaining"], ascending=[True, True, False])
    q3_first_scrimmage = pbp_sorted[(pbp_sorted["qtr"] >= 3) & (pbp_sorted["down"].notna())]
    second_half_recv = (
        q3_first_scrimmage.groupby("game_id", as_index=False)
        .first()[["game_id","posteam"]]
        .rename(columns={"posteam":"second_half_receiver"})
    )
    pbp = pbp.merge(second_half_recv, on="game_id", how="left")
    pbp["receive_2h_ko"] = (pbp["qtr"] <= 2) & (pbp["posteam"] == pbp["second_half_receiver"])

    # Filter to 4th down, scrimmage plays and remove penalties and no-plays
    mask = (
        # 4th down only
        (pbp["down"] == 4)
        # scimmage play
        & (pbp["season_type"].isin(["REG","POST"]))
    )
    # Save state before decision
    state = pbp.loc[mask, KEEP_COLS + ["receive_2h_ko"]].copy()

    state = normalize_cats(state)

    # Select final features before decision
    state = state[[
        "season","game_id","play_id","quarter",
        "game_seconds_remaining","half_seconds_remaining",
        "yardline_100","ydstogo","score_differential",
        "receive_2h_ko","timeouts_off","timeouts_def",
        "roof","surface","temp_f","wind_mph","spread_line",
        "down","season_type","posteam","defteam","home_team","away_team","week"
    ]]

    # Drop rows with important data missing 
    state = state.dropna(subset=["yardline_100","ydstogo","quarter","game_seconds_remaining"])

    outp = OUT_PATH
    state.to_csv(outp, index=False)
    print(f"Saved features before decision in {outp} with {len(state):,} 4th-down plays.")

if __name__ == "__main__":
    main()
