# Web demo (Phase 10)

A small Flask site: enter a developer's profile and the final model predicts their yearly pay (USD, at 2026 levels).

## Files

| File | What it does |
|---|---|
| `src/salary_intel/salary_pipeline.py` | Phases 3, 5 and 6 copied from `notebooks/CURRENTmodeling.ipynb`, wrapped in one object (`SalaryPreprocessor`). Training and the web app both use it, so a new person gets exactly the same steps as the training data. |
| `src/salary_intel/train_final.py` | Refits the rescaled elastic net (Run A settings: alpha = 0.00139, l1_ratio = 0.95) on **all years** and saves `results/final_model.joblib`. |
| `app/app.py` | Flask server. `GET /` shows the form; `POST /predict` returns the prediction. |
| `app/templates/index.html` | The page. |

## Run it

From the repo root:

```bash
pip install -e .                                # installs the salary_intel package + dependencies
python -m salary_intel.train_final --check      # optional: should print 522 columns and val RMSE 0.6783
python -m salary_intel.train_final              # writes results/final_model.joblib
python app/app.py                               # open http://127.0.0.1:5000
```

On macOS, port 5000 is often taken by AirPlay Receiver (you get a 403). Use another port instead:
`python -m flask --app app/app.py run --port 5050`.

Needs `combined/all_years_fixed.csv` (made by `notebooks/MASTERcleanfixMergedData_v2.ipynb`). `results/` is ignored by git; run `train_final` to make the model. Retrain after moving or renaming `salary_pipeline.py`: the saved model remembers the module path of `SalaryPreprocessor`.

## How a prediction is made

1. The form becomes a one-row table that looks like a 2026 survey answer (`survey_year = 2026`).
2. Blank answers get the same labels as the training data ("Not asked" or "No answer"); blank numbers get the training median plus a "was missing" flag.
3. Professional coding years are estimated from work experience, years coding and age with the same regression used for real 2026 answers (the 2026 survey didn't ask for them).
4. Squared columns, log company size, 0/1 list columns, one-hot encoding and scaling are applied with the settings saved at training time.
5. Prediction = exp(Xt · coef + b0). The range shown covers the middle 50% of typical errors (±0.6745 × Run A's 2026 test RMSE, 0.782, on the log scale).
6. "What moved the estimate" adds up coef × (value − training average) for each question's columns, so the bars show how each answer moves pay compared with an average respondent.
