"""Roadmap step 3: SHAP explanations for the best model (LightGBM, detrended, Run A).

    python -m salary_intel.explain          (~10 min, mostly SHAP on 22k rows)

SHAP gives each feature a number for each person, and the numbers add up exactly:

    tree output = base value + sum of the feature contributions

The model is "detrended": a straight line handles the year, the trees the rest, so

    predicted log salary = year line(survey_year) + tree output

The target is log salary, so a contribution c multiplies the predicted salary by
e^c: +0.69 means about x2, -0.69 about /2. SHAP says what the model relies on, not
what causes higher pay, and related features (Age / YearsCode / YearsCodePro) share
credit between them.

shap.TreeExplainer passes LightGBM models to LightGBM's own exact TreeSHAP
(booster.predict(..., pred_contrib=True)). So the checks below test OUR code
(right model, right features, grouping, year line), not LightGBM's math.

Outputs: results/shap/ (CSVs + all plots, ignored by git); a few chosen plots are
copied to docs/figures/ for the README.
"""
import shutil
import warnings

import catboost
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")                       # draw to files, no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from salary_intel.evaluate import metrics  # noqa: E402
from salary_intel.salary_pipeline import CAT_COLS, LIST_COLS  # noqa: E402
from salary_intel.train_trees import CATBOOST, RESULTS, ROOT, YearTrend, load_run  # noqa: E402

OUT = RESULTS / "shap"
FIGURES = ROOT / "docs" / "figures"
RUN, TRAIN_SAMPLE, SEED = "A", 10_000, 42
EXPECTED_TEST_RMSE = 0.68278          # README / model_comparison_A.csv, lgbm_detrended
NICE = {"Country_grouped": "Country", "LanguageHaveWorkedWith": "Languages used",
        "LanguageWantToWorkWith": "Languages wanted", "LanguageHaveWorkedWith_count":
        "Number of languages", "YearsCodePro": "Years coding professionally",
        "YearsCode": "Years coding", "OrgSize": "Company size", "EdLevel": "Education",
        "RemoteWork": "Remote work", "MainBranch": "Developer by profession?",
        "PurchaseInfluence": "Purchasing influence", "DevType": "Job type",
        "LearnCode": "How you learned", "OpSys": "Operating system",
        "Employment": "Employment type", "survey_year": "Survey year (trees)"}


def check(name, worst, tol=1e-6):
    print(f"check {name}: largest difference {worst:.2e}")
    if worst > tol:
        raise SystemExit(f"check failed: {name}")


def groups(pre):
    """Map each model column to the question it came from.

    The 0/1 columns of a multi-answer list (e.g. 144 LanguageHaveWorkedWith_*)
    all map to that list; Age_imputed (a 0/1 flag) maps to Age. SHAP values add
    up, so summing a group's columns gives that question's contribution.
    """
    owner = {raw: lc for lc, cols in pre.list_columns.items() for raw in cols}
    owner["Age_imputed"] = "Age"
    return pd.Series([owner.get(raw, raw) for raw in pre.raw_names], index=pre.columns)


def group_values(values, columns, grp):
    """Per-person contributions, columns -> questions (sum within each group)."""
    return pd.DataFrame(values, columns=columns).T.groupby(grp).sum().T


def numeric_view(F):
    """Feature values for colouring shap plots: numbers stay; categories -> NaN (grey)."""
    N = F.copy()
    for c in CAT_COLS:
        N[c] = np.nan
    return N.astype(float)


def display_view(F, grp):
    """Readable value per question, for the waterfall labels."""
    out = {c: F[c].astype(object).where(F[c].notna(), "unseen") for c in CAT_COLS}
    for c in ["survey_year", "Age", "YearsCode", "YearsCodePro", "OrgSize",
              "LanguageHaveWorkedWith_count"]:
        out[c] = F[c].astype(object).where(F[c].notna(), "blank")
    for c in LIST_COLS:                                   # how many 0/1 columns are 1
        out[c] = F[grp.index[grp == c]].sum(1).map(lambda n: f"{n} ticked")
    return pd.DataFrame(out, index=F.index)


def explain_rows(booster, F):
    exp = shap.TreeExplainer(booster)(F)
    check("additivity (base + contributions = model output)",
          np.abs(exp.values.sum(1) + exp.base_values - booster.predict(F)).max())
    return exp


def importance(G):
    return G.abs().mean().sort_values(ascending=False)


def save(fig_name, docs=False):
    path = OUT / fig_name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    if docs:
        FIGURES.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, FIGURES / fig_name)


def country_plot(G, F, top=15):
    """Average Country contribution per country, as a salary factor (x1.0 = average)."""
    country = F["Country_grouped"].astype(object).fillna("unseen in training").to_numpy()
    stats = pd.DataFrame({"c": country, "v": G["Country_grouped"].to_numpy()}).groupby("c").v \
        .agg(["mean", "size"]).nlargest(top, "size").sort_values("mean")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(stats.index, np.exp(stats["mean"]) - 1, left=1, color="#4C78A8")
    ax.axvline(1, color="grey", lw=0.8)
    ax.set_xlabel("Country's effect on predicted salary (x1.0 = no effect)")
    ax.set_title(f"Country, 2026 test set ({top} most common)")
    for i, (m, n) in enumerate(zip(stats["mean"], stats["size"])):
        ax.text(max(np.exp(m), 1), i, f"  x{np.exp(m):.2f} (n={n})", va="center", fontsize=8)
    return stats


def person_table(i, G, base, trend_term, mean_y, X_te, y_te, pred):
    """One person's explanation: start at the training average, add each part."""
    g = G.iloc[i].sort_values(key=abs, ascending=False)
    parts = pd.Series({"year trend (line, 2026 vs training average)": trend_term[i] - mean_y,
                       "tree base value": base}).pipe(lambda s: pd.concat([s, g]))
    t = pd.DataFrame({"log_contribution": parts, "factor": np.exp(parts)})
    total = mean_y + parts.sum()
    if abs(total - pred[i]) > 1e-6:
        raise SystemExit("check failed: person table doesn't add up to the prediction")
    t.attrs = dict(start_usd=np.exp(mean_y), predicted_usd=np.exp(total),
                   actual_usd=np.exp(y_te.iloc[i]))
    return t


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_run(RUN)
    (X_tr, _, X_te), (F_tr, _, F_te), (y_tr, _, y_te), pre = d["X"], d["F"], d["y"], d["pre"]
    trend = YearTrend().fit(F_tr["survey_year"], y_tr)
    booster = lgb.Booster(model_file=str(RESULTS / f"lgbm_detrended_{RUN}.txt"))

    # 1. Is this the model that was scored?  Same features + same file -> same RMSE.
    trend_te = trend(F_te["survey_year"])
    pred_te = booster.predict(F_te) + trend_te
    check("test RMSE matches step 2", abs(metrics(y_te, pred_te)["rmse_log"] - EXPECTED_TEST_RMSE),
          tol=1e-5)

    # 2. SHAP on the whole 2026 test set and a sample of training rows.
    grp = groups(pre)
    exp_te = explain_rows(booster, F_te)
    sample = F_tr.sample(TRAIN_SAMPLE, random_state=SEED)
    G_te = group_values(exp_te.values, F_te.columns, grp)
    G_tr = group_values(explain_rows(booster, sample).values, F_te.columns, grp)
    check("grouping keeps each person's total", np.abs(G_te.sum(1) - exp_te.values.sum(1)).max())

    # 3. Same question for CatBoost (detrended, Run A): does it rely on the same things?
    cb = catboost.CatBoostRegressor().load_model(str(RESULTS / f"catboost_detrended_{RUN}.cbm"))
    C_te = CATBOOST.prepare(F_te)
    pool = catboost.Pool(C_te, cat_features=[c for c in C_te.columns
                                             if not pd.api.types.is_numeric_dtype(C_te[c])])
    cb_vals = cb.get_feature_importance(pool, type="ShapValues")
    check("CatBoost additivity", np.abs(cb_vals.sum(1) - cb.predict(C_te)).max())
    G_cb = group_values(cb_vals[:, :-1], C_te.columns, grp)

    imp = pd.DataFrame({"lgbm_test": importance(G_te), "lgbm_train_sample": importance(G_tr),
                        "catboost_test": importance(G_cb)})
    imp["typical_factor_lgbm_test"] = np.exp(imp["lgbm_test"])
    imp.index.name = "question"
    imp.to_csv(OUT / "importance_grouped.csv")
    cols = pd.Series(np.abs(exp_te.values).mean(0), index=pre.raw_names).sort_values(ascending=False)
    cols.rename("mean_abs_shap").to_frame().head(40).to_csv(OUT / "importance_columns_top40.csv")
    ranks = imp[["lgbm_test", "lgbm_train_sample", "catboost_test"]].rank(ascending=False)
    print("\nmean |SHAP| by question (log units; 0.10 = about +-10%):")
    print(imp.round(4).to_string())
    print("rank agreement (Spearman): train vs test",
          round(ranks.lgbm_test.corr(ranks.lgbm_train_sample), 3),
          "| LightGBM vs CatBoost", round(ranks.lgbm_test.corr(ranks.catboost_test), 3))

    # 4. Plots.
    names = [NICE.get(g, g) for g in G_te.columns]
    shap.plots.bar(shap.Explanation(G_te.values, feature_names=names), max_display=18, show=False)
    plt.title("What the model relies on most (2026 test, mean |SHAP|, log salary)")
    save("importance_grouped.png", docs=True)

    N_te = numeric_view(F_te)
    exp_num = shap.Explanation(exp_te.values, exp_te.base_values, data=N_te.values,
                               feature_names=pre.raw_names)
    shap.plots.beeswarm(exp_num, max_display=20, show=False)
    plt.title("Top 20 single columns (each dot = one 2026 respondent; grey = category)")
    save("beeswarm_columns.png", docs=True)

    for c in ["YearsCodePro", "Age", "OrgSize"]:
        shap.plots.scatter(exp_num[:, pre.raw_names.index(c)], show=False)
        plt.ylabel(f"SHAP value for {c} (log salary)")
        save(f"effect_{c}.png", docs=(c == "YearsCodePro"))
    stats = country_plot(G_te, F_te)
    save("effect_Country.png", docs=True)
    stats.assign(factor=np.exp(stats["mean"])).to_csv(OUT / "country_effects.csv")

    # 5. Three people, picked by rule: nearest the 25th/50th/75th percentile prediction.
    mean_y, base = float(y_tr.mean()), float(np.ravel(exp_te.base_values)[0])
    show = display_view(F_te, grp)
    tables = []
    for q in (25, 50, 75):
        i = int(np.abs(pred_te - np.percentile(pred_te, q)).argmin())
        t = person_table(i, G_te, base, trend_te, mean_y, X_te, y_te, pred_te)
        a = t.attrs
        print(f"\np{q} person: start ${a['start_usd']:,.0f} (training average) -> predicted "
              f"${a['predicted_usd']:,.0f}, actual ${a['actual_usd']:,.0f}")
        print(t.head(8).round(3).to_string())
        tables.append(t.assign(person=f"p{q}", predicted_usd=a["predicted_usd"],
                               actual_usd=a["actual_usd"]))
        labels = ["Year trend (line)"] + [NICE.get(g, g) for g in G_te.columns]
        data = np.array(["2026"] + [show.iloc[i].get(g, "") for g in G_te.columns], dtype=object)
        w = shap.Explanation(np.r_[trend_te[i] - mean_y, G_te.iloc[i].values],
                             base_values=mean_y + base, data=data, feature_names=labels)
        shap.plots.waterfall(w, max_display=12, show=False)
        plt.title(f"Why this prediction: \\${a['predicted_usd']:,.0f} "      # \$: not math mode
                  f"(actual \\${a['actual_usd']:,.0f}); log salary scale")
        save(f"waterfall_p{q}.png", docs=(q == 50))
    pd.concat(tables).rename_axis("part").to_csv(OUT / "examples.csv")
    print("\nsaved", OUT, "and", FIGURES)


if __name__ == "__main__":
    main()
