"""
Get nflfastR play-by-play and saves NFL seasons play-by-play into Parquet files to data/raw/.
Usage:
  python -m nfl_4thdown_dwp.dataset --seasons 2016-2024
"""
import argparse, os, re
import pandas as pd
import nfl_data_py as nfl

# turn (starting year - ending year) into a and b
def parse_seasons(spec: str):
    if "-" in spec:
        a,b = spec.split("-")
        return list(range(int(a), int(b)+1))
    return [int(x) for x in re.split(r"[,\s]+", spec) if x]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", required=True, help="e.g., 2016-2024")
    ap.add_argument("--outdir", default="data/raw")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    years = parse_seasons(args.seasons)
    print(f"Importing play-by-play for seasons: {years}")

    df = nfl.import_pbp_data(years)
    # Save into Parquets
    out_parq = os.path.join(args.outdir, f"pbp_{years[0]}_{years[-1]}.parquet")
    df.to_parquet(out_parq, index=False)
    print(f"Saved at {out_parq} with {len(df):,} rows.")

if __name__ == "__main__":
    main()
