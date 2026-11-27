"""Exploration (EDA) charts for the Stack Overflow survey, reusable across years.
 
Use on the CLEANED table from clean_survey(). You can refer to columns by a
COMMON NAME (e.g. "education") and the right column is picked for that year.
 
Quick start:
    from salary_intel.explore_survey import *
    run_eda(2019)                          # every standard chart for one year
    compare_salary_years([2017, 2018, 2019, 2020, 2021])   # salary table across years
"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
from salary_intel.clean_survey import clean_survey, parse_years
 
BLUE, GREY, RED = "#2a7ab8", "#c9ced6", "#d1495b"
 
# ---------------------------------------------------------------------------
# Common name -> actual column in each year (None = not asked that year)
# ---------------------------------------------------------------------------
COLUMNS = {
    "country":    {2017: "Country", 2018: "Country", 2019: "Country", 2020: "Country", 2021: "Country"},
    "employment": {2017: "EmploymentStatus", 2018: "Employment", 2019: "Employment",
                   2020: "Employment", 2021: "Employment"},
    "education":  {2017: "FormalEducation", 2018: "FormalEducation", 2019: "EdLevel",
                   2020: "EdLevel", 2021: "EdLevel"},
    "org_size":   {2017: "CompanySize", 2018: "CompanySize", 2019: "OrgSize",
                   2020: "OrgSize", 2021: "OrgSize"},
    "years_code": {2017: "YearsProgram", 2018: "YearsCoding", 2019: "YearsCode",
                   2020: "YearsCode", 2021: "YearsCode"},
    "years_pro":  {2017: "YearsCodedJob", 2018: "YearsCodingProf", 2019: "YearsCodePro",
                   2020: "YearsCodePro", 2021: "YearsCodePro"},
    "dev_type":   {2017: "DeveloperType", 2018: "DevType", 2019: "DevType",
                   2020: "DevType", 2021: "DevType"},
    "languages":  {2017: "HaveWorkedLanguage", 2018: "LanguageWorkedWith", 2019: "LanguageWorkedWith",
                   2020: "LanguageWorkedWith", 2021: "LanguageHaveWorkedWith"},
    "gender":     {2017: "Gender", 2018: "Gender", 2019: "Gender", 2020: "Gender", 2021: "Gender"},
    "age":        {2017: None, 2018: "Age", 2019: "Age", 2020: "Age", 2021: "Age"},
    "job_sat":    {2017: "JobSatisfaction", 2018: "JobSatisfaction", 2019: "JobSat",
                   2020: "JobSat", 2021: None},
}
 
ID_COLS = ["Respondent", "ResponseId"]
RAW_SALARY_COLS = ["Salary", "ExpectedSalary", "ConvertedSalary", "CompTotal",
                   "ConvertedComp", "ConvertedCompYearly"]
ADDED_COLS = ["year", "uid", "salary_usd", "has_salary", "n_answered"]
NON_ANSWERS = ["I prefer not to answer", "I prefer not to say", "I don't know"]
 
 
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _year(df):
    return int(df["year"].iloc[0]) if "year" in df and len(df) else None
 
 
def resolve(df, name):
    """Common name ("education") or real column name -> real column, or None."""
    if name in df.columns:
        return name
    if name in COLUMNS:
        col = COLUMNS[name].get(_year(df))
        if col is None or col not in df.columns:
            print(f"'{name}' was not asked in {_year(df)}; skipping.")
            return None
        return col
    print(f"No column or common name '{name}' in {_year(df)}. "
          f"Common names: {list(COLUMNS)}")
    return None
 
 
def _is_multi(series):
    return series.dropna().astype(str).str.contains(";", regex=False).mean() > 0.02
 
 
def _order_key(value):
    """Sort key from the first number in an answer ("1,000 to 4,999 employees" -> 1000).
    "Less than…" / "Fewer than…" come first."""
    text = str(value).lower().replace(",", "")
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    if text.startswith("just me"):
        return 0.0
    if text.startswith(("less than", "fewer than", "under")):
        return float(nums[0]) - 0.5 if nums else -1.0
    return float(nums[0]) if nums else None
 
 
def _smart_order(values):
    """Order answers by the number in them ("2 to 3 years", "10 to 19 employees").
    Returns None if most answers have no number (then sort by median instead)."""
    vals = list(values)
    keys = [_order_key(v) for v in vals]
    if sum(k is not None for k in keys) >= 0.8 * len(vals):
        return [v for _, v in sorted(zip([k if k is not None else 1e12 for k in keys], vals))]
    return None
 
 
def load_year(year, data_dir="data", keep_all=False):
    """Read the raw CSV for a year and clean it (prints the cleaning log)."""
    raw = pd.read_csv(f"{data_dir}/{year}/results.csv", low_memory=False)
    return clean_survey(raw, year, drop_no_salary=not keep_all)
 
 
# ---------------------------------------------------------------------------
# Missing data
# ---------------------------------------------------------------------------
def plot_salary_sparsity(df_all, group="employment"):
    """Who answered salary. Needs ALL rows: load_year(year, keep_all=True)."""
    col = resolve(df_all, group)
    if col is None:
        return
    if "has_salary" not in df_all:
        print("This table only has salary rows. Use load_year(year, keep_all=True).")
        return
    year, answered = _year(df_all), df_all["has_salary"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [1, 2.4]})
    counts = answered.map({True: "Answered", False: "Blank"}).value_counts().reindex(["Answered", "Blank"], fill_value=0)
    bars = ax1.bar(counts.index, counts.values, color=[BLUE, GREY])
    for b, v in zip(bars, counts.values):
        ax1.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({v/len(df_all):.0%})", ha="center", va="bottom", fontsize=9)
    ax1.set_title(f"{year}: salary answered vs blank"); ax1.set_ylabel("Number of people")
    ax1.set_ylim(0, counts.max() * 1.2)
 
    groups = df_all[col].fillna("(blank)").astype(str)
    tab = pd.crosstab(groups, answered).reindex(columns=[True, False], fill_value=0)
    tab = tab.loc[tab.sum(axis=1).sort_values().index].tail(15)
    ax2.barh(tab.index, tab[True], color=BLUE, label="Answered")
    ax2.barh(tab.index, tab[False], left=tab[True], color=GREY, label="Blank")
    for i, (a, b) in enumerate(zip(tab[True], tab[False])):
        ax2.text(a + b, i, f"  {a/(a+b):.0%} answered", va="center", fontsize=9)
    ax2.set_title(f"{year}: salary by {col} (15 biggest groups)"); ax2.set_xlabel("Number of people")
    ax2.set_xlim(0, tab.sum(axis=1).max() * 1.3); ax2.legend(loc="lower right", frameon=False)
    plt.tight_layout(); plt.show()
 
 
def plot_answer_spread(df_all):
    """How many questions each person answered (best with keep_all=True)."""
    year, n = _year(df_all), df_all["n_answered"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.hist(n, bins=range(0, int(n.max()) + 2), color=BLUE)
    med = n.median()
    ax.axvline(med, color=RED, ls="--", lw=1.5)
    ax.text(med, ax.get_ylim()[1] * 0.95, f" median = {med:.0f} ", color=RED, va="top",
            bbox=dict(facecolor="white", edgecolor="none"))
    ax.set_title(f"{year}: questions answered per person")
    ax.set_xlabel("Questions answered"); ax.set_ylabel("Number of people")
    plt.tight_layout(); plt.show()
    print(f"{year}: {len(n):,} people | median {med:.0f} | under 20 answered: {(n < 20).sum():,}")
 
 
def plot_missing_by_column(df, top=30):
    """% blank for each column (most-blank first)."""
    skip = ADDED_COLS + ID_COLS
    miss = df.drop(columns=[c for c in skip if c in df]).isna().mean().sort_values(ascending=False)
    shown = miss.head(top).sort_values()
    fig, ax = plt.subplots(figsize=(9, 0.28 * len(shown) + 1.5))
    ax.barh(shown.index, shown.values * 100, color=[RED if v > 0.5 else BLUE for v in shown.values])
    ax.set_xlabel("% blank"); ax.set_xlim(0, 100)
    ax.set_title(f"{_year(df)}: most-blank columns (red = over 50% blank)")
    plt.tight_layout(); plt.show()
    print(f"{_year(df)}: {(miss > 0.5).sum()} of {len(miss)} columns are over 50% blank")
 
 
# ---------------------------------------------------------------------------
# Salary
# ---------------------------------------------------------------------------
def salary_summary(df, salary_col="salary_usd"):
    """One row of salary warning signs for this year (issues #2 and #10)."""
    s = df[salary_col].dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    return pd.Series({
        "n": len(s),
        "median": s.median(),
        "p1": s.quantile(0.01),
        "p99": s.quantile(0.99),
        "max": s.max(),
        "people_at_max": int((s == s.max()).sum()),
        "zero": int((s == 0).sum()),
        "under_1k": int((s < 1_000).sum()),
        "over_500k": int((s > 500_000).sum()),
        "over_1M": int((s > 1_000_000).sum()),
        "iqr_upper_fence": q3 + 1.5 * (q3 - q1),
    }, name=_year(df))
 
 
def compare_salary_years(years, data_dir="data"):
    """Salary warning signs side by side for several years."""
    rows = [salary_summary(load_year(y, data_dir)) for y in years]
    table = pd.DataFrame(rows)
    money = ["median", "p1", "p99", "max", "iqr_upper_fence"]
    return table.style.format({c: "${:,.0f}" for c in money} | {c: "{:,.0f}" for c in table.columns if c not in money})
 
 
def plot_salary_distribution(df, salary_col="salary_usd"):
    """Normal histogram, log histogram, box plot."""
    year, s = _year(df), df[salary_col].dropna()
    q1, q3 = s.quantile([0.25, 0.75]); high_fence = q3 + 1.5 * (q3 - q1)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.2))
    ax1.hist(s, bins=60, color=BLUE)
    ax1.axvline(s.median(), color=RED, ls="--", label=f"median \\${s.median():,.0f}")
    ax1.set_title(f"{year}: salary (normal scale)"); ax1.set_xlabel("Salary (USD)")
    ax1.set_ylabel("Number of people"); ax1.legend(frameon=False)
    pos = s[s > 0]
    ax2.hist(pos, bins=np.logspace(np.log10(pos.min()), np.log10(pos.max()), 60), color=BLUE)
    ax2.set_xscale("log")
    for x in [1_000, 200_000, 1_000_000]:
        ax2.axvline(x, color="#888", ls=":")
    ax2.set_title(f"{year}: salary (log scale, \\$0 hidden)\ndotted: \\$1k, \\$200k, \\$1M")
    ax2.set_xlabel("Salary (USD, log scale)")
    ax3.boxplot(s, orientation="horizontal", widths=0.5, flierprops=dict(marker=".", alpha=0.3))
    ax3.axvline(high_fence, color=RED, ls="--", label=f"upper fence \\${high_fence:,.0f}")
    ax3.set_title(f"{year}: box plot"); ax3.set_xlabel("Salary (USD)"); ax3.set_yticks([])
    ax3.legend(frameon=False, loc="upper right")
    plt.tight_layout(); plt.show()
    print(salary_summary(df).to_string())
 
 
def plot_salary_by(df, feature, salary_col="salary_usd", min_salary=None, max_salary=None,
                   order=None, country=None, top=20, ax=None):
    """Salary box plots for each answer to one question.
 
    feature: common name ("education", "years_pro", "org_size"...) or a real column.
    Ranges like "2 to 3 years" are put in numeric order automatically.
    """
    col = resolve(df, feature)
    if col is None:
        return
    base = df if country is None else df[df[resolve(df, "country")] == country]
    data = base[base[salary_col].notna() & base[col].notna()]
    data = data[~data[col].astype(str).isin(NON_ANSWERS)]
    if _is_multi(data[col]):
        print(f"{col} is multi-select; use explore_multiselect() instead.")
        return
    n_before = len(data)
    if min_salary is not None:
        data = data[data[salary_col] >= min_salary]
    if max_salary is not None:
        data = data[data[salary_col] <= max_salary]
    dropped = n_before - len(data)
 
    if order is None:
        top_vals = data[col].value_counts().head(top).index
        order = _smart_order(top_vals) or (
            data[data[col].isin(top_vals)].groupby(col)[salary_col].median().sort_values().index)
    order = [o for o in order if o in set(data[col])]
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 0.4 * len(order) + 1.8))
    groups = [data.loc[data[col] == o, salary_col] for o in order]
    ax.boxplot(groups, orientation="horizontal", showfliers=False, widths=0.6,
               medianprops=dict(color=RED, linewidth=2))
    ax.set_yticks(range(1, len(order) + 1), [f"{o}  (n={len(g):,})" for o, g in zip(order, groups)])
    title = f"{_year(df)}: salary by {col} ({country or 'all countries'})"
    if dropped:
        lo = f"\\${min_salary:,}" if min_salary is not None else "\\$0"
        hi = f"\\${max_salary:,}" if max_salary is not None else "max"
        title += f"\nExcludes {dropped:,} salaries outside {lo} to {hi}"
    ax.set_title(title, fontsize=11); ax.set_xlabel("Salary (USD)")
    ax.xaxis.set_major_formatter(lambda x, _: f"\\${x/1000:.0f}k")
    plt.tight_layout()
    return ax
 
 
# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------
def plot_counts(df, feature, top=20, min_n=50, multi=None):
    """How many people gave each answer. Multi-select is detected automatically."""
    col = resolve(df, feature)
    if col is None:
        return
    s = df[col].dropna().astype(str)
    multi = _is_multi(s) if multi is None else multi
    if multi:
        s = s.str.split(";").explode().str.strip()
    counts = s.value_counts()
    shown = counts.head(top)
    order = _smart_order(shown.index)
    shown = shown.reindex(order[::-1]) if order else shown.sort_values()
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(shown) + 1.6))
    ax.barh(shown.index, shown.values, color=[GREY if v < min_n else BLUE for v in shown.values])
    for i, v in enumerate(shown.values):
        ax.text(v, i, f" {v:,} ({v/counts.sum():.1%})", va="center", fontsize=8.5)
    ax.axvline(min_n, color=RED, ls="--", lw=1)
    ax.set_xlim(0, shown.max() * 1.25)
    extra = f" | top {top} of {len(counts)}" if len(counts) > top else ""
    ax.set_title(f"{_year(df)}: {col}{' (multi-select)' if multi else ''}{extra}\n"
                 f"Grey = fewer than {min_n}", fontsize=11)
    ax.set_xlabel("Number of people")
    plt.tight_layout(); plt.show()
    print(f"{col}: {len(counts)} categories | {(counts < min_n).sum()} have fewer than {min_n} | "
          f"blank: {df[col].isna().mean():.0%}")
 
 
def explore_multiselect(df, feature, top=20, sep=";"):
    """Chart a multi-select question; returns its 0/1 table."""
    col = resolve(df, feature)
    if col is None:
        return
    answered = df[col].notna()
    dummies = df.loc[answered, col].astype(str).str.get_dummies(sep=sep)
    dummies.columns = dummies.columns.str.strip()
    dummies = dummies.T.groupby(level=0).max().T
    n_picked = dummies.sum(axis=1)
    share = dummies.mean().sort_values(ascending=False)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.3, 1]})
    shown = share.head(top).sort_values()
    ax1.barh(shown.index, shown.values * 100, color=BLUE)
    for i, v in enumerate(shown.values):
        ax1.text(v * 100, i, f" {v:.0%}", va="center", fontsize=8.5)
    ax1.set_xlabel(f"% of the {answered.sum():,} people who answered")
    ax1.set_title(f"{_year(df)}: {col}: top {min(top, len(share))} of {len(share)} options")
    ax1.set_xlim(0, shown.max() * 100 * 1.15)
    ax2.hist(n_picked, bins=range(1, dummies.shape[1] + 2), color=BLUE)
    ax2.axvline(n_picked.median(), color=RED, ls="--", label=f"median = {n_picked.median():.0f}")
    ax2.set_yscale("log"); ax2.legend(frameon=False)
    ax2.set_xlabel("Number of options ticked"); ax2.set_ylabel("Number of people (log scale)")
    ax2.set_title(f"{col}: options ticked per person")
    plt.tight_layout(); plt.show()
    print(f"{col}: {answered.mean():.0%} answered | {dummies.shape[1]} options | "
          f"ticked ALL: {(n_picked == dummies.shape[1]).sum()}")
    return dummies
 
 
# ---------------------------------------------------------------------------
# Numeric columns
# ---------------------------------------------------------------------------
def numeric_columns(df):
    """Numeric columns, minus IDs, raw salary columns and columns added by cleaning."""
    return [c for c in df.select_dtypes("number").columns
            if c not in ADDED_COLS + ID_COLS + RAW_SALARY_COLS]
 
 
def plot_numeric(df, cols=None, max_cols=8):
    """One distribution chart per numeric column."""
    if cols is None:
        cols = numeric_columns(df)
        if len(cols) > max_cols:
            print(f"{len(cols)} numeric columns; showing {max_cols}. Pass cols=[...] to choose.\nAll: {cols}")
            cols = cols[:max_cols]
    if not cols:
        print(f"{_year(df)}: no numeric columns besides salary "
              "(experience stays text unless convert_years=True).")
        return
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 3.6), squeeze=False)
    for ax, c in zip(axes[0], cols):
        s = df[c].dropna()
        if s.nunique() <= 41:
            vals = s.value_counts().sort_index()
            ax.bar(vals.index, vals.values, color=BLUE, width=0.8)
        else:
            ax.hist(s, bins=40, color=BLUE)
        ax.set_title(f"{c}\nn={len(s):,} | {s.min():g} to {s.max():g}", fontsize=10)
    axes[0][0].set_ylabel("Number of people")
    plt.tight_layout(); plt.show()
 
 
def plot_correlation(df, cols=None, method="spearman"):
    """Correlation matrix of numeric columns (plus salary_usd)."""
    if cols is None:
        cols = numeric_columns(df)[:10] + ["salary_usd"]
    if len(cols) < 2:
        print(f"{_year(df)}: need at least 2 numeric columns for a correlation matrix.")
        return
    corr = df[cols].corr(method=method)
    fig, ax = plt.subplots(figsize=(1.3 * len(cols) + 2, 1.1 * len(cols) + 1.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=35, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(v) > 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label=f"{method.title()} correlation")
    ax.set_title(f"{_year(df)}: {method.title()} correlation", fontsize=10)
    plt.tight_layout(); plt.show()
 
 
# ---------------------------------------------------------------------------
# Everything at once
# ---------------------------------------------------------------------------
def run_eda(year, data_dir="data", sections=("missing", "salary", "features", "categories", "numeric")):
    """Run the standard EDA for one year. Pick parts with sections=(...)."""
    df_all = load_year(year, data_dir, keep_all=True)
    df = df_all[df_all["has_salary"]].drop(columns="has_salary")
 
    if "missing" in sections:
        print(f"\n===== {year}: MISSING DATA =====")
        plot_salary_sparsity(df_all, "employment")
        plot_answer_spread(df_all)
        plot_missing_by_column(df)
    if "salary" in sections:
        print(f"\n===== {year}: SALARY =====")
        plot_salary_distribution(df)
    if "features" in sections:
        print(f"\n===== {year}: SALARY BY FEATURE (salaries $1k to $1M only) =====")
        for f in ["years_pro", "education", "org_size"]:
            if plot_salary_by(df, f, min_salary=1_000, max_salary=1_000_000) is not None:
                plt.show()
    if "categories" in sections:
        print(f"\n===== {year}: CATEGORIES =====")
        for f in ["country", "gender"]:
            plot_counts(df, f)
        explore_multiselect(df, "languages")
    if "numeric" in sections:
        print(f"\n===== {year}: NUMERIC COLUMNS =====")
        plot_numeric(df)
        plot_correlation(df)
    return df
 
