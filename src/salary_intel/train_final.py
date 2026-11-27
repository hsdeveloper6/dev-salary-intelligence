"""
Phase 10, step 1: build results/final_model.joblib for the web demo.

Run from the repo root (after `pip install -e .`):

    python -m salary_intel.train_final --check   # reproduce Run A first (~30 s)
    python -m salary_intel.train_final           # refit on ALL years and save the model

--check fits the shared preprocessor on 2011-2024, trains the chosen elastic
net, and scores 2025.  If it prints about the same column count and
validation RMSE as the notebook (Phase 7: 0.6783 rescaled), the front end is
using the same steps as the notebook.  It never touches the 2026 test set.
"""
import argparse
import os
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.linear_model import ElasticNet

from salary_intel.salary_pipeline import (CAT_COLS, LIST_COLS, SalaryPreprocessor,
                                          load_csv, phase3, rescale)

# This file is <repo>/src/salary_intel/train_final.py, so parents[2] is <repo>.
ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "combined" / "all_years_fixed.csv"
RESULTS = ROOT / "results"
# Run A's frozen enet_rescaled choice (Phase 7 -> phase9.ipynb): alpha = 0.00139, l1_ratio = 0.95.
# 0.00139 is grid point np.logspace(0, -4, 15)[10]; use the exact value.
A_ALPHA, A_L1 = float(np.logspace(0, -4, 15)[10]), 0.95
VAL_RMSE_A = 0.6783    # Run A validation RMSE (2025), what --check should reproduce
TEST_RMSE_A = 0.78228  # Run A test RMSE (2026, phase9.ipynb): honest error for a new year


def fit_enet(Xt, y):
    m = ElasticNet(alpha=A_ALPHA, l1_ratio=A_L1, max_iter=5000, tol=1e-4)
    return m.fit(sparse.csc_matrix(Xt), y)    # CSC once: avoids a copy inside every fit


def check_run_a(X, y):
    tr, va = X["survey_year"] <= 2024, X["survey_year"] == 2025
    pre = SalaryPreprocessor()
    Xt_tr = pre.fit_transform(X[tr].copy())
    Xt_va = pre.transform(X[va].copy())
    m = fit_enet(Xt_tr, y[tr])
    coef, b0, _ = rescale(m, Xt_tr, y[tr])
    rmse = float(np.sqrt(np.mean((y[va].values - (Xt_va @ coef + b0)) ** 2)))
    print(f"columns: {Xt_tr.shape[1]}  (notebook: 522)")
    print(f"Run A validation RMSE, rescaled EN: {rmse:.4f}  (notebook: {VAL_RMSE_A})")
    print(f"nonzero coefficients: {(coef != 0).sum()}")


def ui_options(X, pre):
    """Choices for the web form, most common first, taken from the training data."""
    hide = {"No answer", "Not asked"}
    opts = {}
    for c in CAT_COLS:
        vc = X[c].value_counts()
        opts[c] = [v for v in vc.index if v not in hide]
    for c in LIST_COLS:
        cols = pre.list_columns[c]
        counts = X[c].str.get_dummies(sep=";").reindex(columns=[k[len(c) + 1:] for k in cols],
                                                       fill_value=0).sum()
        opts[c] = [v for v in counts.sort_values(ascending=False).index if v not in hide]
    org = X["OrgSize"].dropna().value_counts().sort_index()
    opts["OrgSize"] = [float(v) for v in org.index]
    return opts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--csv", default=CSV)
    args = ap.parse_args()

    X, y = phase3(load_csv(args.csv))
    print("rows:", len(X))
    if args.check:
        check_run_a(X, y)
        return

    pre = SalaryPreprocessor()
    Xt = pre.fit_transform(X.copy())            # ALL years, 2011-2026
    m = fit_enet(Xt, y)
    coef, b0, xbar = rescale(m, Xt, y)
    print(f"final model: {Xt.shape[1]} columns, {(coef != 0).sum()} nonzero, "
          f"converged in {m.n_iter_} iterations")

    os.makedirs(RESULTS, exist_ok=True)
    joblib.dump({"pre": pre, "coef": coef, "b0": b0, "xbar": xbar,
                 "names": pre.feature_names(), "alpha": A_ALPHA, "l1_ratio": A_L1,
                 "err_rmse": TEST_RMSE_A, "options": ui_options(X, pre),
                 "defaults": {c: X[c].mode()[0] for c in ["Country_grouped"]},
                 "y_mean": float(np.mean(y))},
                RESULTS / "final_model.joblib", compress=3)
    print("saved", RESULTS / "final_model.joblib")


if __name__ == "__main__":
    main()
