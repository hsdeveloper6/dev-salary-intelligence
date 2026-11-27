"""
Phase 10, step 2: the web demo (input -> model -> output).

    python app/app.py      (from the repo root) then open http://127.0.0.1:5000

The page sends the form as JSON to /predict.  The server builds a one-row
table that looks like a 2026 survey answer, runs it through the SAME fitted
preprocessor saved by train_final.py, and returns the predicted salary plus
which answers pushed it up or down.
"""
import numpy as np
import pandas as pd
import joblib
from scipy import sparse
from flask import Flask, jsonify, render_template, request

from pathlib import Path

from salary_intel.salary_pipeline import (CAT_COLS, GROUP_LABELS, FEATURES, LIST_COLS,
                                          feature_group, label_blanks)

# This file is <repo>/app/app.py, so parents[1] is <repo>; train_final.py saves there.
ROOT = Path(__file__).resolve().parents[1]
B = joblib.load(ROOT / "results" / "final_model.joblib")
PRE, COEF, B0, XBAR = B["pre"], B["coef"], B["b0"], B["xbar"]
GROUPS = np.array([feature_group(n) for n in B["names"]])
Z50 = 0.6745                        # half of a normal curve lies within +-0.6745 sd

app = Flask(__name__)


def num(v, lo, hi):
    """Blank -> NaN (the model fills it with the training median and flags it)."""
    if v in (None, ""):
        return np.nan
    v = float(v)
    if not lo <= v <= hi:
        raise ValueError(f"{v} is outside {lo}-{hi}")
    return v


def build_row(f):
    langs = f.get("LanguageHaveWorkedWith") or []
    row = {
        "survey_year": 2026,                       # predict at current pay levels
        "Age": num(f.get("Age"), 14, 90),
        "YearsCode": num(f.get("YearsCode"), 0, 60),
        "YearsCodePro": np.nan,                    # 2026 survey didn't ask it: imputed
        "WorkExp": num(f.get("WorkExp"), 0, 60),   # from WorkExp, like real 2026 rows
        "OrgSize": num(f.get("OrgSize"), 1, 1e6),
        "LanguageHaveWorkedWith_count": len(langs) if langs else np.nan,
    }
    for c in CAT_COLS:
        row[c] = f.get(c) or np.nan
    for c in LIST_COLS:
        picks = f.get(c) or []
        row[c] = ";".join(picks) if picks else np.nan
    df = pd.DataFrame([row])[FEATURES]
    df[CAT_COLS + LIST_COLS] = df[CAT_COLS + LIST_COLS].astype(object)  # blank text stays text
    return label_blanks(df)


@app.get("/")
def index():
    return render_template("index.html", options=B["options"],
                           defaults=B["defaults"], labels=GROUP_LABELS)


@app.post("/predict")
def predict():
    try:
        X = build_row(request.get_json(force=True))
    except ValueError as e:
        return jsonify(error=str(e)), 400

    xt = PRE.transform(X)
    xt = np.asarray(xt.toarray() if sparse.issparse(xt) else xt).ravel()
    log_pred = float(xt @ COEF + B0)

    # Each column's push compared with the average respondent in the data:
    # prediction = mean log salary + sum of coef_j * (x_j - mean_j)
    contrib = pd.Series(COEF * (xt - XBAR)).groupby(GROUPS).sum()
    contrib = contrib[contrib.abs() > 1e-4].sort_values(key=abs, ascending=False)
    factors = [{"label": GROUP_LABELS.get(k, k),
                "pct": round(float(np.expm1(v)) * 100, 1)} for k, v in contrib.items()]

    # Show the estimated professional years so the demo can explain Phase 5
    ycp = float(PRE._phase5(X)["YearsCodePro"].iloc[0])

    half = Z50 * B["err_rmse"]   # Run A test RMSE on 2026
    return jsonify(
        salary=round(float(np.exp(log_pred)), -2),
        low=round(float(np.exp(log_pred - half)), -2),
        high=round(float(np.exp(log_pred + half)), -2),
        average=round(float(np.exp(B["y_mean"])), -2),
        yearsCodePro=None if np.isnan(ycp) else round(ycp, 1),
        factors=factors[:8])


if __name__ == "__main__":
    app.run(debug=False)
