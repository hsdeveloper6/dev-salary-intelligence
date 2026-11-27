"""Rebuild data/harmonized/{year}_droppedcols.csv for 2017-2021 from the raw survey files.

    python -m salary_intel.build_harmonized        # run from the repo root

Two steps per year, the same ones the team ran by hand:
  1. raw data/<year>/results.csv -> clean_survey() -> data/clean/{year}_basic.csv
     (was the loop in the class repo's basic_clean_and_save_data.ipynb)
  2. data/clean/{year}_basic.csv -> harmonize()   -> data/harmonized/{year}_droppedcols.csv
     (was notebooks/dropcols.ipynb)

The output matches the files the team originally committed byte for byte. Step 2
reads step 1's CSV back instead of using the in-memory table: reading a CSV
re-guesses column types, and the original files were made that way, so skipping
the round trip could change the output.
"""
from pathlib import Path

import pandas as pd

from salary_intel.clean_survey import clean_survey
from salary_intel.harmonize import harmonize

# This file is <repo>/src/salary_intel/build_harmonized.py, so parents[2] is <repo>.
ROOT = Path(__file__).resolve().parents[2]
YEARS = [2017, 2018, 2019, 2020, 2021]


def build(year, root=ROOT):
    raw = pd.read_csv(root / f"data/{year}/results.csv", low_memory=False)
    basic = root / f"data/clean/{year}_basic.csv"
    clean_survey(raw, year).to_csv(basic, index=False)

    h = harmonize(pd.read_csv(basic, low_memory=False), year)
    out = root / f"data/harmonized/{year}_droppedcols.csv"
    h.to_csv(out, index=False)
    print(f"saved {out.relative_to(root)}: {h.shape[0]:,} rows, {h.shape[1]} columns\n")


def main():
    (ROOT / "data/clean").mkdir(parents=True, exist_ok=True)
    (ROOT / "data/harmonized").mkdir(parents=True, exist_ok=True)
    for year in YEARS:
        build(year)


if __name__ == "__main__":
    main()
