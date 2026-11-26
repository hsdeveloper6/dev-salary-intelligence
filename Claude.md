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
- After any change to the pipeline, run `python train_final.py --check` and
  confirm it still gives 522 columns and validation RMSE ~0.6783.
- Commit after each working step with a clear message.

## Data
- The modeling dataset is `combined/all_years_fixed.csv` (386,701 rows). It is
  on my machine only, NOT in git. Never commit data files or `results/*.joblib`.
- Load it with `pd.read_csv(path, keep_default_na=False, na_values=[""])` so real
  answers like "None"/"NA" stay as text.
- Raw survey: 803,388 responses; cleaning removed students/unemployed/hobbyists
  and salaries outside $1,000-$1,000,000.

## Current pipeline (phase10_frontend/salary_pipeline.py is the clean version)
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
- Current best model: elastic net (alpha ~0.00139, l1_ratio 0.95, rescaled per
  Zou & Hastie 2005).
- Test results: Run A R^2 0.42, median error ~$28.6k, RMSE 0.782 (log).
  Run B R^2 0.61, median error ~$14.1k.

## Roadmap (do in this order)
1. Restructure repo (src/, app/, notebooks/, archive/), stop tracking CSVs,
   write README with credit line: "Extends a team project from UC Davis ECS 171
   (Fall 2026), which I led."
2. LightGBM (`train_trees.py`) on the same splits and metrics; CatBoost as a
   comparison. Compare against elastic net fairly.
3. SHAP explanations (global importance + per-prediction).
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
overlaps the language 0/1 columns.
