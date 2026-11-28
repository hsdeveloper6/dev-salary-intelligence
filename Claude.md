# Developer Salary Intelligence

## What this is
A portfolio project: predicting developer salaries from 16 years (2011-2026) of
Stack Overflow Developer Survey data. It extends a 3-person UC Davis ECS 171
(Fall 2026) team project that I led; I'm now continuing it independently to put
on my resume and talk about in interviews.

## How to work with me
- I'm a CS student still learning. Explain what you're doing and why, in plain
  language, especially for anything ML or deployment related.
- Work in small steps. Plan big changes before making them.
- I need to understand every line I keep (interviewers will ask), so explain
  non-obvious code and point out trade-offs.
- After any change to the pipeline, run (from the repo root)
  `python -m salary_intel.train_final --check` and confirm it still gives 522
  columns and validation RMSE ~0.6783. (Needs `pip install -e .` once.)
- Commit after each working step with a clear message.

## Data
- The modeling dataset is `combined/all_years_fixed.csv` (386,701 rows). It is
  on my machine only, NOT in git. Never commit data files or `results/*.joblib`.
  No CSVs are tracked: the three merge inputs are rebuilt from the raw files
  with `python -m salary_intel.build_premerge` (2011-2016),
  `build_harmonized` (2017-2021) and `build_cleaned_v2` (2022-2026).
- Load it with `pd.read_csv(path, keep_default_na=False, na_values=[""])` so real
  answers like "None"/"NA" stay as text.
- Raw survey: 803,388 responses; cleaning removed students/unemployed/hobbyists
  and salaries outside $1,000-$1,000,000.

## Current pipeline (src/salary_intel/salary_pipeline.py is the clean version)
- Target: `y = log(Comp_USD_Annual)`.
- 19 input features; categories use "Not asked" (question absent that year) or
  "No answer" (skipped).
- `survey_year` is a plain number, never one-hot.
- Phase 5 imputation: Age for 2017 (donor from 2016/2018), YearsCodePro via
  regression (missing 2011-2016 and 2025-2026). WorkExp dropped after.
- Phase 6: squared terms, log10 OrgSize, 0/1 columns for multi-answer lists,
  one-hot for single-answer categories, StandardScaler -> 522 columns.
- Everything that learns is fit on training data only (no leakage).
- Splits: Run A (time-based) train 2011-2024, validate 2025, test 2026.
  Run B (random 70/15/15, stratified by year, random_state=42).
- Elastic net (alpha ~0.00139, l1_ratio 0.95, rescaled per Zou & Hastie 2005):
  Run A test R^2 0.42, median error ~$28.6k, RMSE 0.782 (log), mean error -0.35;
  Run B R^2 0.61, median error ~$14.1k. Reference: `python -m salary_intel.evaluate`.
- Best so far: LightGBM (`src/salary_intel/train_trees.py`), frozen settings in
  `results/lgbm_params_{A,B}.json`:
  Run A test (detrended) R^2 0.56, median error ~$20.6k, RMSE 0.683, mean error +0.06
  (plain: R^2 0.54, ~$21.0k, RMSE 0.693, mean error -0.13).
  Run B R^2 0.65, median error ~$13.2k, RMSE 0.621 (variants identical).
  The variants tie on 2025 validation; the web demo still uses the elastic net.
- CatBoost (same script, `--model catboost`, `results/catboost_params_{A,B}.json`)
  is behind LightGBM on both runs: Run A test detrended RMSE 0.698, R^2 0.54,
  ~$20.9k (plain 0.715, mean error -0.18); Run B RMSE 0.626, ~$13.4k. It got 14
  trials per variant at learning rate 0.15 (vs LightGBM 60 at 0.05); a second,
  widened round moved validation by <=0.0002.
- Tree features drop the `YearsCodePro_imputed` flag: it is a pure era marker
  (1 for 2011-2016 and 2025-2026, ~0 otherwise).

## Roadmap (do in this order)
1. Restructure repo (src/, app/, notebooks/, archive/), stop tracking CSVs,
   write README with credit line: "Extends a team project from UC Davis ECS 171
   (Fall 2026), which I led."
1b. Make the merge inputs reproducible (after the restructure):
    - [done] 2011-2016: src/salary_intel/build_premerge.py runs the cleaning
      notebooks (raw -> data/clean/canonical_{year}.csv; matches
      archive/clean_dataset/ and regenerates the missing 2016 file), then a
      keep/rename table -> data/premerge_cleaned_dataset/. Byte-identical to the
      old tracked files; now untracked. The table was traced from the committed
      files, not built from the spreadsheet (its 2011-2016 columns hold raw
      headers, so it couldn't be cross-checked automatically).
    - [done] 2017-2021: src/salary_intel/build_harmonized.py (clean_survey ->
      data/clean/{year}_basic.csv -> harmonize -> data/harmonized/) reproduces
      the old tracked files byte for byte; data/harmonized/ is now untracked.
    - [done] 2022-2026: src/salary_intel/build_cleaned_v2.py (clean_survey ->
      data/clean/survey_{year}_cleaned.csv -> keep/rename table -> data/cleaned_v2/)
      reproduces the old tracked files byte for byte; data/cleaned_v2/ is now
      untracked. It keeps the old step's spreadsheet deviations on purpose
      (2023 AIAcc/AIBen swap, 11 unrenamed 2026 columns; marked "# sheet:").
      Fixing them is a later change, because it changes the merged data.
    Each is done when its output matches the tracked files exactly. Once a
    folder can be fully recreated by code, it can be untracked.
2. LightGBM (`train_trees.py`) on the same splits and metrics; CatBoost as a
   comparison. Compare against elastic net fairly.
   [done: LightGBM and CatBoost, plain + detrended variants, all beat the
   elastic net on both runs; LightGBM is best.]
3. SHAP explanations (global importance + per-prediction).
   [done: src/salary_intel/explain.py on LightGBM detrended, Run A test. Country
   dominates (mean |SHAP| 0.48), then YearsCodePro (0.18), OrgSize (0.08); CatBoost
   agrees (Spearman 0.94). Found `OpSys_Not asked` = era marker (OS not asked in
   2017 + 2026), -0.08 on every 2026 prediction; fix = retrain, not done yet.]
4. Prediction intervals: quantile regression and/or conformal prediction;
   check coverage on 2026.
5. Drift analysis: train up to year N, test on N+1, plot error by year.
6. Deploy: FastAPI + Docker, host publicly; add MLflow, tests, GitHub Actions.

## Traps for tree models
- Trees can't extrapolate `survey_year`: they treat 2026 like 2024 (no wage
  growth). Try inflation-adjusting salaries (CPI) or letting a linear model
  handle the year trend.
- Keep the Phase 5 YearsCodePro imputation. If left blank, trees learn
  "blank = 2011-2016 respondent" and apply it to all 2025-2026 rows.
- Trees don't need squared terms, scaling, or one-hot for single-answer
  categories (use pandas `category` dtype, with categories fixed from training).
  Multi-answer lists still need 0/1 columns.

## Known limitations
Nominal (not inflation-adjusted) salaries; salary bands 2011-2014; survey not a
random sample; ~11% of 2026 Industry answers unseen in training; language count
overlaps the language 0/1 columns; OpSys "Not asked" only in 2017 and 2026 (era
marker for trees).
