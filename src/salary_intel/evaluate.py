"""Shared evaluation code: the two splits, the metrics, and the elastic-net reference.

Every model (elastic net, LightGBM, later CatBoost) is scored with these same
functions, so comparisons are fair:
  * make_splits() reproduces the notebooks' Run A / Run B rows exactly.
  * metrics() is phase9.ipynb's scoring (cell 5), plus a mean-error column.
  * elastic_net_reference() refits the frozen Phase 7 elastic net and must
    reproduce phase9's test numbers. That proves the splits and metrics match.
"""
import numpy as np
from scipy import sparse
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from salary_intel.salary_pipeline import SalaryPreprocessor, rescale

# Frozen Phase 7 choices (enet_rescaled). Both alphas are points on the
# notebooks' grid np.logspace(0, -4, 15); use the exact values, not rounded ones.
ENET = {
    "A": dict(alpha=float(np.logspace(0, -4, 15)[10]), l1_ratio=0.95, max_iter=5000),   # 0.00139
    "B": dict(alpha=float(np.logspace(0, -4, 15)[11]), l1_ratio=0.99, max_iter=10000),  # 0.00072
}
# phase9.ipynb test results, which elastic_net_reference() must reproduce
PHASE9_TEST = {"A": dict(rmse=0.78228, r2=0.41942, median_abs_err=28563.53),
               "B": dict(rmse=0.65379, r2=0.61326, median_abs_err=14103.88)}


def make_splits(X, y, run):
    """Return (X_train, X_val, X_test, y_train, y_val, y_test), as in the notebooks.

    Run A: train 2011-2024, validate 2025, test 2026 (CURRENTmodeling.ipynb).
    Run B: random 70/15/15, stratified by year, random_state=42
           (V3_RUNB_CURRENTmodeling.ipynb cell 22: test split first, then validation).
    """
    if run == "A":
        tr, va, te = (X["survey_year"] <= 2024, X["survey_year"] == 2025,
                      X["survey_year"] == 2026)
        return X[tr].copy(), X[va].copy(), X[te].copy(), y[tr], y[va], y[te]
    if run == "B":
        X_tmp, X_test, y_tmp, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=X["survey_year"])
        X_train, X_val, y_train, y_val = train_test_split(
            X_tmp, y_tmp, test_size=0.15 / 0.85, random_state=42,
            stratify=X_tmp["survey_year"])
        return X_train, X_val, X_test, y_train, y_val, y_test
    raise ValueError(f"run must be 'A' or 'B', got {run!r}")


def metrics(y_true_log, y_pred_log):
    """phase9's metrics, plus mean error (predicted - actual, in log units).

    A negative mean error means the model guesses too low on average. That is
    how missing wage growth (the tree year trap) shows up.
    """
    t = np.asarray(y_true_log, dtype=float).ravel()
    p = np.asarray(y_pred_log, dtype=float).ravel()
    return {
        "rmse_log": float(np.sqrt(mean_squared_error(t, p))),
        "r2_log": float(r2_score(t, p)),
        "median_abs_err_usd": float(median_absolute_error(np.exp(t), np.exp(p))),
        "mean_err_log": float(np.mean(p - t)),
    }


def elastic_net_reference(X_train, X_val, X_test, y_train, run):
    """Refit the frozen Phase 7 elastic net; return its (val, test) log predictions.

    The preprocessor is fit on training rows only, then applied to validation
    and then test, in that order (the order matters: 2017 age imputation draws
    from one random-number stream).
    """
    cfg = ENET[run]
    pre = SalaryPreprocessor()
    Xt_tr = pre.fit_transform(X_train.copy())
    Xt_va = pre.transform(X_val.copy())
    Xt_te = pre.transform(X_test.copy())
    model = ElasticNet(alpha=cfg["alpha"], l1_ratio=cfg["l1_ratio"],
                       max_iter=cfg["max_iter"], tol=1e-4)
    model.fit(sparse.csc_matrix(Xt_tr), y_train)
    coef, b0, _ = rescale(model, Xt_tr, y_train)
    return np.asarray(Xt_va @ coef).ravel() + b0, np.asarray(Xt_te @ coef).ravel() + b0


def main():
    """python -m salary_intel.evaluate: check the reference against phase9 (~30 s).

    Test RMSE should match to 5 decimals. The median $ error can differ by a few
    dollars: the notebook refit with warm_start, this refits from scratch, and
    the two stop at slightly different points within the solver's tolerance.
    """
    from salary_intel.salary_pipeline import load_csv, phase3
    from salary_intel.train_final import CSV

    X, y = phase3(load_csv(CSV))
    for run in ("A", "B"):
        X_tr, X_va, X_te, y_tr, y_va, y_te = make_splits(X, y, run)
        p_va, p_te = elastic_net_reference(X_tr, X_va, X_te, y_tr, run)
        v, t, ref = metrics(y_va, p_va), metrics(y_te, p_te), PHASE9_TEST[run]
        print(f"Run {run}: val RMSE {v['rmse_log']:.4f} | test RMSE {t['rmse_log']:.5f} "
              f"(phase9 {ref['rmse']}) | R2 {t['r2_log']:.5f} (phase9 {ref['r2']}) | "
              f"median $ {t['median_abs_err_usd']:,.0f} (phase9 {ref['median_abs_err']:,.0f}) | "
              f"mean err {t['mean_err_log']:+.4f}")


if __name__ == "__main__":
    main()
