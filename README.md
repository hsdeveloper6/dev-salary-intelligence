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
  ├─ 2011-2016  python -m salary_intel.build_premerge
  │                 notebooks/cleaning/clean_20XX_canonical ──> data/clean/canonical_20XX.csv
  │                 keep + rename ──> data/premerge_cleaned_dataset/final_canonical_20XX.csv
  ├─ 2017-2021  python -m salary_intel.build_harmonized
  │                 clean_survey ──> data/clean/{year}_basic.csv
  │                 harmonize    ──> data/harmonized/{year}_droppedcols.csv
  └─ 2022-2026  python -m salary_intel.build_cleaned_v2
                    clean_survey      ──> data/clean/survey_{year}_cleaned.csv
                    keep + rename     ──> data/cleaned_v2/survey_{year}_v2.csv
                         │
        notebooks/combinemerge ──> combined/all_years.csv
        notebooks/MASTERcleanfixMergedData_v2 ──> combined/all_years_fixed.csv
                         │
        notebooks/CURRENTmodeling (Run A), V3_RUNB_CURRENTmodeling (Run B),
        phase8, phase9  ──>  src/salary_intel/salary_pipeline.py + train_final.py
```

### Rebuilding the merge inputs

No data is tracked in git. The three merge inputs that `combinemerge.ipynb`
reads are rebuilt from the raw survey files, each one byte for byte identical
to the files the team originally committed:

```bash
python -m salary_intel.download_data       # raw files -> data/<year>/
python -m salary_intel.build_premerge      # 2011-2016 -> data/premerge_cleaned_dataset/  (~1 min)
python -m salary_intel.build_harmonized    # 2017-2021 -> data/harmonized/
python -m salary_intel.build_cleaned_v2    # 2022-2026 -> data/cleaned_v2/
```

Then run `notebooks/combinemerge.ipynb` and
`notebooks/MASTERcleanfixMergedData_v2.ipynb` to make `combined/all_years_fixed.csv`.

`build_premerge` runs the 2011-2016 cleaning notebooks in order, in a temporary
folder: later notebooks read alias tables that earlier ones write.

`build_cleaned_v2` reproduces the old step exactly, including places where it
disagrees with the schema spreadsheet. In 2023, `AIAcc`/`AIBen` are swapped
compared with 2024 (Stack Overflow swapped the raw names that year). In 2026,
11 columns keep their raw names (e.g. `Education`, `LanguagesHaveWorkedWith`)
instead of the canonical ones. Each rule is marked `# sheet:` in the module.

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
  build_premerge.py    rebuilds data/premerge_cleaned_dataset/ (2011-2016) from the raw files
  build_harmonized.py  rebuilds data/harmonized/ (2017-2021) from the raw files
  build_cleaned_v2.py  rebuilds data/cleaned_v2/ (2022-2026) from the raw files
  clean_survey.py, harmonize.py, explore_survey.py, download_data.py
app/                Flask web demo (see app/README.md)
notebooks/          modeling, merge and EDA notebooks; cleaning/ = per-year cleaning + EDA
docs/               schema mapping spreadsheet
data/               raw downloads and everything rebuilt from them (ignored)
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
