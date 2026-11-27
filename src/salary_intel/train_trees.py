"""Roadmap step 2: tree models (LightGBM, CatBoost) on the same splits and metrics as the elastic net.

    python -m salary_intel.train_trees --run A --model lgbm --tune [--trials 30] [--append]
        Validation only. Random search, then saves results/{model}_params_A.json
        and results/{model}_trials_A.csv. The test set is not touched.
    python -m salary_intel.train_trees --run A --test
        Scores the elastic net and every model that has frozen settings, on validation
        and test, and saves results/model_comparison_A.csv. Each tree model is refit on
        the training rows with its frozen settings.

Two variants per model, because trees can't extrapolate survey_year (they predict
2026 like 2024, missing wage growth):
  plain      target = log salary; survey_year is an ordinary feature.
  detrended  a straight line  log salary ~ a + b * survey_year  is fit on the
             TRAINING rows only. The trees learn the leftover (residual), and
             prediction = line(year) + trees. The line carries the growth into
             future years; the trees model everything else.

Each model type is a small "backend" (LGBM, CATBOOST below). It knows its fixed
settings, its search grid, and how to fit, predict and save. Everything else is shared.
"""
import argparse
import json
import time
from pathlib import Path

import catboost
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
MAX_TREES, EARLY_STOP = 5000, 100


class LGBM:
    name = "lgbm"
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

    @staticmethod
    def prepare(F):
        return F                                   # LightGBM reads pandas categories directly

    @classmethod
    def fit(cls, params, F_tr, t_tr, F_va=None, t_va=None, n_trees=MAX_TREES):
        """Returns (model, number of trees used)."""
        train = lgb.Dataset(F_tr, t_tr, free_raw_data=False)
        kw = {}
        if F_va is not None:
            kw = dict(valid_sets=[lgb.Dataset(F_va, t_va, reference=train)],
                      callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        m = lgb.train({**cls.FIXED, **params}, train, num_boost_round=n_trees, **kw)
        return m, (m.best_iteration if F_va is not None else n_trees)

    @staticmethod
    def predict(m, F, n_trees):
        return m.predict(F, num_iteration=n_trees)

    @staticmethod
    def save(m, path):
        m.save_model(str(path.with_suffix(".txt")))


class CATBOOST:
    name = "catboost"
    FIXED = dict(loss_function="RMSE", learning_rate=0.08, random_seed=42,
                 thread_count=8, verbose=0, allow_writing_files=False)
    SPACE = dict(depth=[4, 6, 8, 10],
                 l2_leaf_reg=[1.0, 3.0, 10.0, 30.0],
                 random_strength=[0.5, 1.0, 2.0],
                 bagging_temperature=[0.0, 0.5, 1.0],
                 border_count=[128, 254])

    @staticmethod
    def prepare(F):
        """CatBoost needs text categories, not pandas categories with NaN.

        In training these columns are never blank ("No answer"/"Not asked" are real
        labels). A blank only appears in validation/test, for an answer never seen
        in training, so it becomes "__unseen__", a category CatBoost has never seen.
        """
        F = F.copy()
        for c in F.columns[F.dtypes == "category"]:
            F[c] = F[c].astype(object).where(F[c].notna(), "__unseen__").astype(str)
        return F

    @classmethod
    def fit(cls, params, F_tr, t_tr, F_va=None, t_va=None, n_trees=MAX_TREES):
        # The text columns made by prepare(). Check "not numeric" rather than
        # dtype == object: in pandas 3, astype(str) gives the new "str" dtype.
        cats = [c for c in F_tr.columns if not pd.api.types.is_numeric_dtype(F_tr[c])]
        m = catboost.CatBoostRegressor(**cls.FIXED, **params, iterations=n_trees,
                                       cat_features=cats)
        if F_va is None:
            m.fit(F_tr, t_tr)
            return m, n_trees
        m.fit(F_tr, t_tr, eval_set=(F_va, t_va), early_stopping_rounds=EARLY_STOP,
              use_best_model=True)
        return m, m.get_best_iteration() + 1      # best_iteration counts from 0

    @staticmethod
    def predict(m, F, n_trees):
        return m.predict(F, ntree_end=n_trees)

    @staticmethod
    def save(m, path):
        m.save_model(str(path.with_suffix(".cbm")))


BACKENDS = {b.name: b for b in (LGBM, CATBOOST)}


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
    """What gets added back to the trees' output: 0 (plain) or the trend line (detrended)."""
    return [np.zeros(len(f)) if variant == "plain" else trend(f["survey_year"]) for f in frames]


def tune(run, backend, trials, append=False):
    """Random search on validation. With append=True, keep the earlier trials in
    results/{model}_trials_{run}.csv, draw new ones with seed 43, and pick the best
    across all of them (every trial is scored on the same validation rows)."""
    d = load_run(run)
    (F_tr, F_va, _), (y_tr, y_va, _) = d["F"], d["y"]
    trend = YearTrend().fit(F_tr["survey_year"], y_tr)
    F_tr, F_va = backend.prepare(F_tr), backend.prepare(F_va)
    trials_path = RESULTS / f"{backend.name}_trials_{run}.csv"
    if append and not trials_path.exists():
        raise SystemExit(f"--append needs {trials_path} from an earlier --tune")
    previous = pd.read_csv(trials_path) if append else None
    rng = np.random.default_rng(43 if append else 42)
    space = backend.SPACE
    draws = [{k: v[rng.integers(len(v))] for k, v in space.items()} for _ in range(trials)]
    draws = [{k: (float(v) if isinstance(v, float) else int(v)) for k, v in p.items()} for p in draws]

    rows = []
    for variant in VARIANTS:
        off_tr, off_va = offsets(variant, trend, [F_tr, F_va])
        for i, params in enumerate(draws):
            t0 = time.time()
            m, n = backend.fit(params, F_tr, y_tr - off_tr, F_va, y_va - off_va)
            v = metrics(y_va, backend.predict(m, F_va, n) + off_va)
            rows.append(dict(variant=variant, search=2 if append else 1, trial=i, **params,
                             n_trees=n, val_rmse=v["rmse_log"], val_mean_err=v["mean_err_log"],
                             seconds=round(time.time() - t0, 1)))
            print(f"[{backend.name} {run} {variant} {i + 1}/{trials}] val RMSE {v['rmse_log']:.4f} "
                  f"trees {n} ({rows[-1]['seconds']}s) {params}", flush=True)

    hist = pd.DataFrame(rows)
    if previous is not None:
        if "search" not in previous:
            previous.insert(1, "search", 1)
        hist = pd.concat([previous, hist], ignore_index=True)
    RESULTS.mkdir(exist_ok=True)
    hist.to_csv(trials_path, index=False)
    frozen = {"run": run, "model": backend.name,
              "trend": {"intercept": float(trend.lr.intercept_),
                        "slope_per_year": float(trend.lr.coef_[0])},
              "fixed": backend.FIXED, "variants": {}}
    for variant in VARIANTS:
        best = hist[hist.variant == variant].sort_values("val_rmse").iloc[0]
        frozen["variants"][variant] = dict(
            params={k: best[k].item() for k in space}, n_trees=int(best.n_trees),
            val_rmse=float(best.val_rmse))
        print(f"best {variant}: val RMSE {best.val_rmse:.4f}, {int(best.n_trees)} trees, "
              f"{frozen['variants'][variant]['params']}")
    path = RESULTS / f"{backend.name}_params_{run}.json"
    path.write_text(json.dumps(frozen, indent=2))
    print("saved", path)


def test(run):
    frozen = {name: json.loads(p.read_text()) for name in BACKENDS
              if (p := RESULTS / f"{name}_params_{run}.json").exists()}
    if not frozen:
        raise SystemExit(f"no results/*_params_{run}.json found: run --tune first. The test "
                         "set is only scored with settings frozen from validation.")
    d = load_run(run)
    (X_tr, X_va, X_te), (F_tr, F_va, F_te), (y_tr, y_va, y_te) = d["X"], d["F"], d["y"]
    trend = YearTrend().fit(F_tr["survey_year"], y_tr)

    preds = {"elastic_net": elastic_net_reference(X_tr, X_va, X_te, y_tr, run)}
    for name, fz in frozen.items():
        b = BACKENDS[name]
        G_tr, G_va, G_te = b.prepare(F_tr), b.prepare(F_va), b.prepare(F_te)
        for variant in VARIANTS:
            cfg = fz["variants"][variant]
            off_tr, off_va, off_te = offsets(variant, trend, [G_tr, G_va, G_te])
            m, n = b.fit(cfg["params"], G_tr, y_tr - off_tr, n_trees=cfg["n_trees"])
            preds[f"{name}_{variant}"] = (b.predict(m, G_va, n) + off_va,
                                          b.predict(m, G_te, n) + off_te)
            b.save(m, RESULTS / f"{name}_{variant}_{run}")

    rows = [dict(model=name, split=split, **metrics(y, p))
            for name, (p_va, p_te) in preds.items()
            for split, y, p in (("val", y_va, p_va), ("test", y_te, p_te))]
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / f"model_comparison_{run}.csv", index=False)
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 160):
        print(f"\nRun {run}\n", table.pivot(index="model", columns="split").swaplevel(axis=1)
              .sort_index(axis=1, level=0))
    print("saved", RESULTS / f"model_comparison_{run}.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["A", "B"], required=True)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tune", action="store_true")
    mode.add_argument("--test", action="store_true")
    ap.add_argument("--model", choices=sorted(BACKENDS), default="lgbm",
                    help="with --tune: which model to search")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--append", action="store_true",
                    help="with --tune: add new trials to the existing ones (seed 43)")
    args = ap.parse_args()
    if args.tune:
        tune(args.run, BACKENDS[args.model], args.trials, args.append)
    else:
        test(args.run)


if __name__ == "__main__":
    main()
