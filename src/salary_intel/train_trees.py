"""Roadmap step 2: LightGBM on the same splits and metrics as the elastic net.

    python -m salary_intel.train_trees --run A --tune [--trials 30]
        Validation only. Random search, then saves results/lgbm_params_A.json
        and results/lgbm_trials_A.csv. The test set is not touched.
    python -m salary_intel.train_trees --run A --test
        Needs the frozen JSON. Refits on training rows, scores the elastic net and
        both LightGBM variants on validation and test, and saves
        results/model_comparison_A.csv.

Two variants, because trees can't extrapolate survey_year (they predict 2026
like 2024, missing wage growth):
  plain      target = log salary; survey_year is an ordinary feature.
  detrended  a straight line  log salary ~ a + b * survey_year  is fit on the
             TRAINING rows only. LightGBM learns the leftover (residual), and
             prediction = line(year) + LightGBM. The line carries the growth into
             future years; the trees model everything else.
"""
import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from salary_intel.evaluate import elastic_net_reference, make_splits, metrics
from salary_intel.salary_pipeline import load_csv, phase3
from salary_intel.train_final import CSV
from salary_intel.tree_features import TreePreprocessor

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
VARIANTS = ["plain", "detrended"]

FIXED = dict(objective="regression", learning_rate=0.05, bagging_freq=1, seed=42,
             deterministic=True, force_col_wise=True, num_threads=8, verbose=-1)
# The first search (seed 42) used num_leaves up to 255, min_child_samples up to 200
# and lambda_l2 up to 10. Its best settings sat at those upper edges, so the grid
# was extended, and a second search (--append, seed 43) added trials on top.
SPACE = dict(num_leaves=[31, 63, 127, 255, 511, 1023],
             min_child_samples=[20, 50, 100, 200, 400, 800],
             feature_fraction=[0.5, 0.7, 0.9],
             bagging_fraction=[0.7, 0.9, 1.0],
             lambda_l2=[0.0, 1.0, 10.0, 30.0, 100.0])
MAX_TREES, EARLY_STOP = 5000, 100


def load_run(run):
    """Splits plus tree features (fit on training rows) for one run."""
    X, y = phase3(load_csv(CSV))
    X_tr, X_va, X_te, y_tr, y_va, y_te = make_splits(X, y, run)
    pre = TreePreprocessor()
    F_tr = pre.fit_transform(X_tr.copy())
    F_va, F_te = pre.transform(X_va.copy()), pre.transform(X_te.copy())
    return dict(X=(X_tr, X_va, X_te), F=(F_tr, F_va, F_te), y=(y_tr, y_va, y_te))


class YearTrend:
    """log salary ~ a + b * survey_year, fit on training rows only."""

    def fit(self, years, y):
        self.lr = LinearRegression().fit(np.asarray(years, float).reshape(-1, 1), y)
        return self

    def __call__(self, years):
        return self.lr.predict(np.asarray(years, float).reshape(-1, 1))


def offsets(variant, trend, frames):
    """What gets added back to LightGBM's output: 0 (plain) or the trend line (detrended)."""
    return [np.zeros(len(f)) if variant == "plain" else trend(f["survey_year"]) for f in frames]


def fit_lgbm(params, F_tr, target_tr, F_va=None, target_va=None, n_trees=MAX_TREES):
    train = lgb.Dataset(F_tr, target_tr, free_raw_data=False)
    kw = {}
    if F_va is not None:
        kw = dict(valid_sets=[lgb.Dataset(F_va, target_va, reference=train)],
                  callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    return lgb.train({**FIXED, **params}, train, num_boost_round=n_trees, **kw)


def tune(run, trials, append=False):
    """Random search on validation. With append=True, keep the earlier trials in
    results/lgbm_trials_{run}.csv, draw new ones with seed 43, and pick the best
    across all of them (every trial is scored on the same validation rows)."""
    d = load_run(run)
    (F_tr, F_va, _), (y_tr, y_va, _) = d["F"], d["y"]
    trend = YearTrend().fit(F_tr["survey_year"], y_tr)
    trials_path = RESULTS / f"lgbm_trials_{run}.csv"
    if append and not trials_path.exists():
        raise SystemExit(f"--append needs {trials_path} from an earlier --tune")
    previous = pd.read_csv(trials_path) if append else None
    rng = np.random.default_rng(43 if append else 42)
    draws = [{k: v[rng.integers(len(v))] for k, v in SPACE.items()} for _ in range(trials)]
    draws = [{k: (float(v) if isinstance(v, float) else int(v)) for k, v in p.items()} for p in draws]

    rows = []
    for variant in VARIANTS:
        off_tr, off_va = offsets(variant, trend, [F_tr, F_va])
        for i, params in enumerate(draws):
            t0 = time.time()
            m = fit_lgbm(params, F_tr, y_tr - off_tr, F_va, y_va - off_va)
            pred = m.predict(F_va, num_iteration=m.best_iteration) + off_va
            v = metrics(y_va, pred)
            rows.append(dict(variant=variant, search=2 if append else 1, trial=i, **params,
                             n_trees=m.best_iteration,
                             val_rmse=v["rmse_log"], val_mean_err=v["mean_err_log"],
                             seconds=round(time.time() - t0, 1)))
            print(f"[{run} {variant} {i + 1}/{trials}] val RMSE {v['rmse_log']:.4f} "
                  f"trees {m.best_iteration} ({rows[-1]['seconds']}s) {params}", flush=True)

    hist = pd.DataFrame(rows)
    if previous is not None:
        if "search" not in previous:
            previous.insert(1, "search", 1)
        hist = pd.concat([previous, hist], ignore_index=True)
    RESULTS.mkdir(exist_ok=True)
    hist.to_csv(trials_path, index=False)
    frozen = {"run": run, "trend": {"intercept": float(trend.lr.intercept_),
                                    "slope_per_year": float(trend.lr.coef_[0])},
              "fixed": FIXED, "variants": {}}
    for variant in VARIANTS:
        best = hist[hist.variant == variant].sort_values("val_rmse").iloc[0]
        frozen["variants"][variant] = dict(
            params={k: best[k].item() for k in SPACE}, n_trees=int(best.n_trees),
            val_rmse=float(best.val_rmse))
        print(f"best {variant}: val RMSE {best.val_rmse:.4f}, {int(best.n_trees)} trees, "
              f"{frozen['variants'][variant]['params']}")
    (RESULTS / f"lgbm_params_{run}.json").write_text(json.dumps(frozen, indent=2))
    print("saved", RESULTS / f"lgbm_params_{run}.json")


def test(run):
    path = RESULTS / f"lgbm_params_{run}.json"
    if not path.exists():
        raise SystemExit(f"{path} not found: run --tune first. The test set is only "
                         "scored with settings frozen from validation.")
    frozen = json.loads(path.read_text())
    d = load_run(run)
    (X_tr, X_va, X_te), (F_tr, F_va, F_te), (y_tr, y_va, y_te) = d["X"], d["F"], d["y"]
    trend = YearTrend().fit(F_tr["survey_year"], y_tr)

    preds = {"elastic_net": elastic_net_reference(X_tr, X_va, X_te, y_tr, run)}
    for variant in VARIANTS:
        cfg = frozen["variants"][variant]
        off_tr, off_va, off_te = offsets(variant, trend, [F_tr, F_va, F_te])
        m = fit_lgbm(cfg["params"], F_tr, y_tr - off_tr, n_trees=cfg["n_trees"])
        preds[f"lgbm_{variant}"] = (m.predict(F_va) + off_va, m.predict(F_te) + off_te)
        m.save_model(str(RESULTS / f"lgbm_{variant}_{run}.txt"))

    rows = [dict(model=name, split=split, **metrics(y, p))
            for name, (p_va, p_te) in preds.items()
            for split, y, p in (("val", y_va, p_va), ("test", y_te, p_te))]
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / f"model_comparison_{run}.csv", index=False)
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 120):
        print(f"\nRun {run}\n", table.pivot(index="model", columns="split").swaplevel(axis=1)
              .sort_index(axis=1, level=0))
    print("saved", RESULTS / f"model_comparison_{run}.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["A", "B"], required=True)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tune", action="store_true")
    mode.add_argument("--test", action="store_true")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--append", action="store_true",
                    help="with --tune: add new trials to the existing ones (seed 43)")
    args = ap.parse_args()
    tune(args.run, args.trials, args.append) if args.tune else test(args.run)


if __name__ == "__main__":
    main()
