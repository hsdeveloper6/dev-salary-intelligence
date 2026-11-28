"""
Phase 10 shared preprocessing: Phases 3, 5 and 6 from CURRENTmodeling.ipynb,
packed into ONE object so training and the web app run identical steps.

The functions below are copied from the pushed notebook (commit c8a7df5,
cells 8-12, 28, 42-48). Changes are limited to:
  * globals the notebook relied on (numeric_imputer, NUMERIC lists) are passed
    in or stored on the SalaryPreprocessor object,
  * audit prints/asserts that only make sense for whole splits are dropped,
  * finalize_numeric() no longer prints clip counts for every web request.
Phase 7's rescale() is copied as well (cell 52).

`fit_transform(X)` learns everything from the rows it is given (training
data); `transform(X)` replays it unchanged on any table, including a
one-row table typed into the web page.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ================================================================ Phase 3
DROP_FOR_MODEL = ["Comp_USD_Annual", "CompTotal", "ConvertedCompYearly", "comp_outlier_country",
                  "Currency", "Currency_ISO", "CompFreq", "comp_source", "uid", "ResponseId",
                  "Country", "Region", "JobSat", "CompIsTotalComp"]

NUMERIC = ["survey_year", "Age", "YearsCode", "YearsCodePro", "WorkExp", "OrgSize",
           "LanguageHaveWorkedWith_count"]

NOT_ASKED = {
    "EdLevel": list(range(2011, 2017)),
    "Employment": list(range(2011, 2015)),
    "Industry": list(range(2017, 2023)),
    "RemoteWork": [2011, 2012, 2013, 2018, 2020, 2021],
    "LearnCode": [2011, 2012, 2013, 2014, 2020],
    "OpSys": [2017, 2026],
    # 2026 asked MainBranch as "CodeMain"; the merge now renames it, so only ~5 blank 2026
    # rows still get "Not asked". 2026 stays listed so the original data reproduces phase9.
    "MainBranch": list(range(2011, 2017)) + [2018, 2026],
    "LanguageWantToWorkWith": [2011, 2012, 2013],
    "PurchaseInfluence": [2016, 2018, 2021],
}
ALWAYS_ASKED = ["DevType", "LanguageHaveWorkedWith", "Country_grouped"]
CATEGORICAL = list(NOT_ASKED) + ALWAYS_ASKED
FEATURES = NUMERIC + CATEGORICAL                     # the 19 predictors


def load_csv(path):
    """Same read as the notebook: only empty cells become NaN."""
    return pd.read_csv(path, keep_default_na=False, na_values=[""], low_memory=False)


def phase3(df):
    """Cells 4, 8, 10, 12: log target, drop non-inputs, label category blanks."""
    y = None
    if "Comp_USD_Annual" in df:
        y = pd.Series(np.log(df["Comp_USD_Annual"].astype("float64").to_numpy()),
                      index=df.index, name="log_annual_salary_usd")
    X = df.drop(columns=[c for c in DROP_FOR_MODEL if c in df.columns]).copy()
    extra = sorted(set(X.columns) - set(FEATURES))
    missing = sorted(set(FEATURES) - set(X.columns))
    if extra or missing:
        raise ValueError(f"Input schema differs. Extra: {extra}; missing: {missing}")
    X = X.loc[:, FEATURES]

    X["survey_year"] = pd.to_numeric(X["survey_year"], errors="raise").astype("int64")
    for col in NUMERIC:
        if col != "survey_year":
            X[col] = pd.to_numeric(X[col], errors="raise")
    return label_blanks(X), y


def label_blanks(X):
    """Cell 12, unchanged."""
    X = X.copy()
    for col in CATEGORICAL:
        values = X[col].astype("string").str.strip().replace("", pd.NA)
        if col in NOT_ASKED:
            missing_not_asked = values.isna() & X["survey_year"].isin(NOT_ASKED[col])
            values = values.mask(missing_not_asked, "Not asked")
        X[col] = values.fillna("No answer")
    return X


# ================================================================ Phase 5 (cell 28)
def fit_age_donors(train):
    mask = (train["survey_year"].isin([2016, 2018])
            & train[["Age", "YearsCode"]].notna().all(axis=1))
    donors = train.loc[mask, ["YearsCode", "Age"]]
    if donors.empty:
        raise ValueError("No eligible training-only age donors.")
    return {code: group["Age"].to_numpy(dtype=float)
            for code, group in donors.groupby(donors["YearsCode"].round())}


def impute_age(data, pools, rng):
    out = data.copy()
    original_missing = out["Age"].isna()
    out["Age_imputed"] = original_missing.astype("int8")
    mask = original_missing & out["survey_year"].eq(2017) & out["YearsCode"].notna()
    donor_keys = np.array(sorted(pools.keys()), dtype=float)
    if len(donor_keys) == 0:
        raise ValueError("No age donor groups available.")
    groups = out.loc[mask].groupby(out.loc[mask, "YearsCode"].round()).groups
    for code, indices in groups.items():
        if code in pools:
            candidates = pools[code]
        else:
            nearby = donor_keys[np.abs(donor_keys - float(code)) <= 2]
            if len(nearby) == 0:
                continue
            candidates = np.concatenate([pools[key] for key in nearby])
        out.loc[indices, "Age"] = rng.choice(candidates, size=len(indices), replace=True)
    return out


def fit_ycp(train):
    specs = {
        "recent": (["WorkExp", "YearsCode", "Age"], train["survey_year"].between(2022, 2024)),
        "early": (["YearsCode", "Age"], train["survey_year"].between(2018, 2024)),
        "backup": (["YearsCode"], train["survey_year"].between(2017, 2024)),
    }
    fitted = {}
    for name, (cols, year_mask) in specs.items():
        rows = train.loc[year_mask].dropna(subset=cols + ["YearsCodePro"])
        if len(rows) <= len(cols) + 10:
            raise ValueError(f"{name}: insufficient complete training rows ({len(rows)})")
        fitted[name] = (cols, LinearRegression().fit(rows[cols], rows["YearsCodePro"]))
    return fitted


def impute_ycp(data, models):
    out = data.copy()
    original_missing = out["YearsCodePro"].isna()
    out["YearsCodePro_imputed"] = original_missing.astype("int8")
    year = out["survey_year"]
    target_years = year.le(2016) | year.isin([2025, 2026])
    steps = [("recent", year.isin([2025, 2026])),
             ("early", year.le(2016)),
             ("backup", target_years)]
    for name, eligible in steps:
        cols, model = models[name]
        mask = out["YearsCodePro"].isna() & eligible & out[cols].notna().all(axis=1)
        if not mask.any():
            continue
        predictions = model.predict(out.loc[mask, cols])
        max_years = out.loc[mask, "YearsCode"].to_numpy(dtype=float)
        out.loc[mask, "YearsCodePro"] = np.clip(predictions, 0, max_years)
    return out


# ================================================================ Phase 6 (cells 42-48)
BASE_NUMERIC = ["Age", "YearsCode", "YearsCodePro", "OrgSize", "LanguageHaveWorkedWith_count"]
NEW_MISSING_FLAGS = {"YearsCode": "YearsCode_missing",
                     "OrgSize": "OrgSize_missing",
                     "LanguageHaveWorkedWith_count": "LanguageHaveWorkedWith_count_missing"}
LIST_COLS = ["DevType", "LearnCode", "LanguageHaveWorkedWith",
             "LanguageWantToWorkWith", "OpSys", "Employment"]
CAT_COLS = ["Country_grouped", "Industry", "EdLevel", "RemoteWork", "MainBranch",
            "PurchaseInfluence"]
NUM_COLS = ["survey_year", "Age", "YearsCode", "YearsCodePro", "LanguageHaveWorkedWith_count",
            "Age_sq", "YearsCode_sq", "YearsCodePro_sq", "OrgSize_log",
            "Age_imputed", "YearsCodePro_imputed",
            "YearsCode_missing", "OrgSize_missing", "LanguageHaveWorkedWith_count_missing"]


def finalize_numeric(data, numeric_imputer):
    """Cell 42 (6a): flags -> training medians -> clip YearsCodePro -> squares -> log10."""
    out = data.copy()
    for col, flag in NEW_MISSING_FLAGS.items():
        out[flag] = out[col].isna().astype("int8")
    out[BASE_NUMERIC] = numeric_imputer.transform(out[BASE_NUMERIC])
    if (out["YearsCode"] < 0).any():
        raise ValueError("Negative YearsCode found.")
    if (out["OrgSize"] <= 0).any():
        raise ValueError("OrgSize must be positive for log10.")
    out["YearsCodePro"] = np.minimum(np.maximum(out["YearsCodePro"].to_numpy(), 0),
                                     out["YearsCode"].to_numpy())
    for col in ["Age", "YearsCode", "YearsCodePro"]:
        out[col + "_sq"] = out[col] ** 2
    out["OrgSize_log"] = np.log10(out["OrgSize"])
    return out.drop(columns=["OrgSize"])


def encode_lists(data, list_columns=None):
    """Cell 44 (6b): 0/1 columns per answer; reindexed to training columns if given."""
    parts = []
    for col in present_lists(data):
        enc = data[col].astype("string").str.get_dummies(sep=";").add_prefix(col + "_")
        if list_columns is not None:
            enc = enc.reindex(columns=list_columns[col], fill_value=0)
        parts.append(enc.astype(pd.SparseDtype("uint8", fill_value=0)))
    return parts


def present_lists(data):
    """The multi-answer columns still in the data (an input can be dropped, e.g. OpSys)."""
    return [c for c in LIST_COLS if c in data.columns]


def assemble(data, parts):
    return pd.concat([data.drop(columns=present_lists(data)), *parts], axis=1)


def make_prep_all():
    """Cells 46-48 (6c, 6d)."""
    prep = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scale", StandardScaler())]), NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CAT_COLS),
        ],
        remainder="passthrough",
        sparse_threshold=1.0,
    )
    return Pipeline([("prep", prep), ("scale_all", StandardScaler(with_mean=False))])


class SalaryPreprocessor:
    """Phases 5-6 fitted once, then replayed. Fit with the notebook's seed (42).

    drop: inputs to leave out entirely, e.g. ("OpSys",). The default () is the
    notebooks' pipeline, which train_final --check reproduces.
    """

    def __init__(self, seed=42, drop=()):
        self.seed = seed
        self.drop = tuple(drop)

    def _phase5(self, X):
        X = X.drop(columns=list(self.drop))
        return impute_ycp(impute_age(X, self.age_pools, self._rng), self.ycp_models) \
            .drop(columns="WorkExp")

    def fit_transform(self, X):
        self._rng = np.random.default_rng(self.seed)
        self.age_pools = fit_age_donors(X)
        self.ycp_models = fit_ycp(X)
        d = self._phase5(X)
        self.numeric_imputer = SimpleImputer(strategy="median").fit(d[BASE_NUMERIC])
        d = finalize_numeric(d, self.numeric_imputer)
        parts = encode_lists(d)
        self.list_columns = {c: p.columns.tolist() for c, p in zip(present_lists(d), parts)}
        d = assemble(d, parts)
        self.prep_all = make_prep_all()
        return self.prep_all.fit_transform(d)

    def transform(self, X):
        d = finalize_numeric(self._phase5(X), self.numeric_imputer)
        return self.prep_all.transform(assemble(d, encode_lists(d, self.list_columns)))

    def feature_names(self):
        return self.prep_all.named_steps["prep"].get_feature_names_out()


def rescale(model, Xt_tr, y_train):
    """Cell 52: beta_rescaled = (1 + lambda2) * beta_naive; intercept recomputed."""
    xbar = np.asarray(Xt_tr.mean(axis=0)).ravel()
    lam2 = float(model.alpha) * (1 - float(model.l1_ratio))
    coef = (1 + lam2) * np.asarray(model.coef_, dtype=float)
    intercept = float(np.mean(y_train)) - xbar @ coef
    return coef, float(intercept), xbar


# ================================================================ explanation
GROUP_LABELS = {
    "survey_year": "Pay level in 2026", "Age": "Age", "YearsCode": "Years coding",
    "YearsCodePro": "Professional experience", "OrgSize": "Company size",
    "LanguageHaveWorkedWith": "Languages used",
    "Country_grouped": "Country", "Industry": "Industry", "EdLevel": "Education",
    "RemoteWork": "Remote work", "MainBranch": "Developer role",
    "PurchaseInfluence": "Purchasing influence", "DevType": "Job type",
    "LearnCode": "How you learned", "LanguageWantToWorkWith": "Languages wanted",
    "OpSys": "Operating system", "Employment": "Employment type"}


def feature_group(name):
    """Map a transformed column name back to the question it came from."""
    raw = name.split("__", 1)[1]
    if raw.startswith("LanguageHaveWorkedWith"):       # count + dummies share credit
        return "LanguageHaveWorkedWith"
    for key in sorted(GROUP_LABELS, key=len, reverse=True):
        if raw == key or raw.startswith(key + "_"):
            return key
    return raw
