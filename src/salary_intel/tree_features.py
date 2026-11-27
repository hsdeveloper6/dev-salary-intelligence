"""Features for tree models (LightGBM now, CatBoost later).

Same idea as SalaryPreprocessor: fit_transform() learns everything from the
training rows, and transform() replays it unchanged on validation/test.
It reuses salary_pipeline.py's Phase 5 and list encoding. It leaves out the
steps that only a linear model needs (see CLAUDE.md, "Traps for tree models"):

  kept   Phase 5 imputation: Age for 2017, YearsCodePro for 2011-2016 and
         2025-2026. Without it, a blank YearsCodePro would mean "2011-2016
         respondent" in training, and trees would wrongly apply that to every
         2025-2026 row.
  kept   0/1 columns for the 6 multi-answer lists (columns fixed from training).
  new    the 6 single-answer questions become pandas "category" columns, with
         the category list fixed from training. An answer never seen in
         training becomes missing, not a new category.
  left   no squared terms, no scaling, no log10(OrgSize): a tree only asks
  out    "is x above some cut-off?", so a curve or a rescaling changes nothing.
         Other blank numbers stay NaN, because LightGBM handles missing values itself.
"""
import re

import numpy as np
import pandas as pd

from salary_intel.salary_pipeline import (CAT_COLS, LIST_COLS, encode_lists, fit_age_donors,
                                          fit_ycp, impute_age, impute_ycp)


def _safe_names(columns):
    """LightGBM rejects some characters in feature names (e.g. "C#", "Bash/Shell (all, shells)").

    Replace anything that isn't a letter, digit or underscore. If two names collide,
    add a number so every name stays unique.
    """
    out, seen = [], {}
    for c in columns:
        s = re.sub(r"[^0-9A-Za-z_]+", "_", str(c)).strip("_")
        n = seen.get(s, 0)
        seen[s] = n + 1
        out.append(s if n == 0 else f"{s}_{n}")
    return out


class TreePreprocessor:
    def __init__(self, seed=42):
        self.seed = seed

    def _phase5(self, X):
        out = impute_ycp(impute_age(X, self.age_pools, self._rng), self.ycp_models)
        return out.drop(columns="WorkExp")

    def _encode(self, d, list_columns=None):
        for c in CAT_COLS:
            d[c] = pd.Categorical(d[c], categories=self.categories[c])
        parts = encode_lists(d, list_columns)
        dense = [p.sparse.to_dense().astype("uint8") for p in parts]
        out = pd.concat([d.drop(columns=LIST_COLS), *dense], axis=1)
        return out, parts

    def fit_transform(self, X):
        self._rng = np.random.default_rng(self.seed)
        self.age_pools = fit_age_donors(X)
        self.ycp_models = fit_ycp(X)
        d = self._phase5(X)
        self.categories = {c: sorted(d[c].dropna().unique()) for c in CAT_COLS}
        out, parts = self._encode(d)
        self.list_columns = {c: p.columns.tolist() for c, p in zip(LIST_COLS, parts)}
        self.raw_names = out.columns.tolist()          # readable names, for SHAP later
        self.columns = _safe_names(self.raw_names)
        out.columns = self.columns
        return out

    def transform(self, X):
        out, _ = self._encode(self._phase5(X), self.list_columns)
        if out.columns.tolist() != self.raw_names:
            raise ValueError("transform produced different columns than training")
        out.columns = self.columns
        return out
