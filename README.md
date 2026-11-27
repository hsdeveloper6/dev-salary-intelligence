# Developer Salary Intelligence

Predicting developer salaries from 16 years (2011-2026) of the
[Stack Overflow Developer Survey](https://survey.stackoverflow.co/).

Extends a team project from UC Davis ECS 171 (Fall 2026), which I led.

## Results

The model is an elastic net on log salary (alpha ≈ 0.00139, l1_ratio 0.95,
rescaled per Zou & Hastie 2005), scored on held-out test data:

| Split | Train / validate / test | Test R² | Median abs. error | RMSE (log) |
|---|---|---|---|---|
| **Run A** (time-based) | 2011-2024 / 2025 / 2026 | 0.42 | ~$28.6k | 0.782 |
| **Run B** (random 70/15/15, stratified by year) | mixed years | 0.61 | ~$14.1k | n/a |

Run A is the honest number for "predict a new year's salaries". Run B is easier:
the test years were also seen in training.

## Data

- **Source:** the official survey files (`python -m salary_intel.download_data`
  fetches them into `data/<year>/`).
- **Size:** 803,388 raw responses. Cleaning removed students, unemployed people
  and hobbyists, and salaries outside $1,000-$1,000,000. The modeling table is
  `combined/all_years_fixed.csv`, with 386,701 rows. It's not in git.
- **Schema mapping:** questions and answer options change from year to year.
  [`docs/SO_Survey_Schema_Harmonization_2011-2026.xlsx`](docs/SO_Survey_Schema_Harmonization_2011-2026.xlsx)
  maps every year's columns to one shared set of names.
  [`harmonize.py`](src/salary_intel/harmonize.py) puts that mapping into code.

### Data flow

```
data/<year>/results.csv  (raw, downloaded)
  ├─ 2011-2016  notebooks/cleaning/clean_20XX_canonical ──> canonical_20XX.csv
  │                 ──(missing step)──> data/premerge_cleaned_dataset/final_canonical_20XX.csv  [committed]
  ├─ 2017-2021  python -m salary_intel.build_harmonized
  │                 clean_survey ──> data/clean/{year}_basic.csv
  │                 harmonize    ──> data/harmonized/{year}_droppedcols.csv
  └─ 2022-2026  (missing step) ──> data/cleaned_v2/survey_20XX_v2.csv  [committed]
                         │
        notebooks/combinemerge ──> combined/all_years.csv
        notebooks/MASTERcleanfixMergedData_v2 ──> combined/all_years_fixed.csv
                         │
        notebooks/CURRENTmodeling (Run A), V3_RUNB_CURRENTmodeling (Run B),
        phase8, phase9  ──>  src/salary_intel/salary_pipeline.py + train_final.py
```

### Committed inputs

These two folders are tracked in git. They were produced outside this repo, and
no code here can recreate them yet:

| Folder | Years | Missing step that produces it |
|---|---|---|
| `data/premerge_cleaned_dataset/` | 2011-2016 | `canonical_20XX.csv` → `final_canonical_20XX.csv` |
| `data/cleaned_v2/` | 2022-2026 | the script that writes `survey_20XX_v2.csv` |

Rebuilding these steps is on the roadmap. Once a folder can be recreated by code,
it gets untracked. `data/harmonized/` (2017-2021) already went through this:
`python -m salary_intel.build_harmonized` rebuilds it from the raw files, byte
for byte identical to the files the team originally committed.

## Pipeline

- **Target:** `log(Comp_USD_Annual)`. Using the log makes errors relative, so being
  $10k off matters more at $40k than at $400k.
- **19 inputs:**
  - Numbers: age, years coding, professional years, company size, language count, survey year.
  - Single-answer categories: country, industry, education, remote work, role,
    purchasing influence.
  - Multi-answer lists: job type, languages used/wanted, OS, employment, how you learned.
- **Missing answers:** a category is labelled "Not asked" if the question didn't
  exist that year, and "No answer" if the person skipped it.
- **Imputation, fit on training data only:**
  - Age for 2017 comes from similar 2016/2018 respondents.
  - Professional years (missing 2011-2016 and 2025-2026) come from a regression on
    years coding, age and work experience.
- **Features:**
  - Squared terms for age and experience, and log10 of company size.
  - 0/1 columns for multi-answer lists, and one-hot columns for single-answer categories.
  - Standard scaling. The result is 522 columns.

Every step that learns something (medians, regressions, encoders, scalers) is fit
on the training split only, then replayed unchanged on validation, test and the
web demo.

## Layout

```
src/salary_intel/   Python package
  salary_pipeline.py   preprocessing (Phases 3, 5, 6) as one SalaryPreprocessor object
  train_final.py       --check reproduces Run A; default trains the demo model
  build_harmonized.py  rebuilds data/harmonized/ (2017-2021) from the raw files
  clean_survey.py, harmonize.py, explore_survey.py, download_data.py
app/                Flask web demo (see app/README.md)
notebooks/          modeling, merge and EDA notebooks; cleaning/ = per-year cleaning + EDA
docs/               schema mapping spreadsheet
data/               raw downloads and rebuilt files (ignored) + the two committed inputs
archive/            superseded notebooks (plus ignored old CSV outputs on disk)
```

## Run it

```bash
pip install -e .                               # installs salary_intel and its dependencies
python -m salary_intel.train_final --check     # 522 columns, Run A validation RMSE 0.6783
python -m salary_intel.train_final             # trains on all years -> results/final_model.joblib
python app/app.py                              # web demo at http://127.0.0.1:5000
```

You need `combined/all_years_fixed.csv` first (see Data flow). Notebooks can be
opened from any folder: their first cell changes to the repo root.

## Known limitations

- Salaries are nominal, not adjusted for inflation.
- 2011-2014 salaries come as bands, not exact numbers.
- The survey is not a random sample of developers.
- About 11% of 2026 Industry answers never appear in training.
- The language count overlaps the language 0/1 columns.
