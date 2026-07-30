"""
Populist Rhetoric Regression Analysis
======================================

Setup
-----
DV : populist rhetoric (%), three dimensions -> anti_elitism, people_centrism, populism
IV : election_year (0/1), pop_seat_share_group (%), pop_seat_share_ep (%)
     [crisis_year deliberately left out for now -- add back in once defined]

Unit of observation for the regressions : party x year (panel / TSCS, 11
parties x 1999-2024).

RAW DATA is sentence-level, one row per sentence:
    sentence_id, speech_id, unique_id, speaker, party, period,
    legislative_year, sentence,
    anti_elitism_binary, people_centrism_binary, populism_combined,
    populist_ep_seats, total_ep_seats, populist_ep_share,
    populist_group_seats, total_group_seats, populist_share_of_group

The functions in section 1 below aggregate this up to party x year
(the actual regression unit) BEFORE anything in section 2 onward runs.

"""
# conda activate ma_ep_populism
# pip install pandas statsmodels linearmodels

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
from statsmodels.stats.outliers_influence import variance_inflation_factor
from linearmodels.panel import PanelOLS
from linearmodels.panel.results import compare
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# DATA STRUCTURE NOTE
# ---------------------------------------------------------------------------
# With 11 parties x 1999-2024 (26 years), this is a LONG PANEL
# (T=26 > N=11) -- i.e. classic time-series-cross-section (TSCS) data as
# used throughout comparative politics. This has two consequences handled
# below:
#
# 1. Cluster-robust SEs by party are unreliable with only 11 clusters
#    (rule of thumb: cluster-robust inference wants ~30-50+ clusters).
#    -> pooled models below use Driscoll-Kraay SEs instead (robust to
#       within-party serial correlation AND cross-party correlation in the
#       same year, e.g. all parties reacting to the same event).
#
# 2. election_year (and crisis_year, once you add it) almost certainly
#    varies only over time (same value for all 11 parties in a given
#    year), not across parties.
#    -> Do NOT combine year fixed effects with these dummies: year FE
#       will absorb them completely. Party (entity) fixed effects are fine
#       and used by default below.

EP_ELECTION_YEARS = [2004, 2009, 2014, 2019, 2024]
EP_CRISIS_YEARS = [2009, 2015, 2022] # Euro area crisis, Refugee crisis, Energy crisis (Russian war)


# ---------------------------------------------------------------------------
# 1. AGGREGATE SENTENCE-LEVEL DATA -> PARTY x YEAR x DIMENSION
# ---------------------------------------------------------------------------

def aggregate_to_party_year(df_sentences,
                             party_col="party",
                             year_col="legislative_year",
                             dim_binary_cols=None,
                             min_sentences=None):
    
    if dim_binary_cols is None:
        dim_binary_cols = {
            "anti_elitism": "anti_elitism_binary",
            "people_centrism": "people_centrism_binary",
            "populism": "populism_combined",
        }


    # --- sanity check: 
    seat_cols = ["populist_ep_share", "populist_share_of_group"]
    n_unique = (
        df_sentences.groupby([party_col, year_col])[seat_cols]
        .nunique()
    )
    inconsistent = n_unique[(n_unique > 1).any(axis=1)]
    if len(inconsistent) > 0:
        print(f"WARNING: {len(inconsistent)} party-year groups have "
              f"non-constant seat-share values. Inspect before proceeding:")
        print(inconsistent)

    # Conversion to long format
    long_frames = []
    for dim_name, col in dim_binary_cols.items():
        grp = df_sentences.groupby([party_col, year_col]).agg(
            rhetoric_pct=(col, lambda x: 100 * x.mean()),
            n_sentences=(col, "size"),
            populist_ep_share=("populist_ep_share", "mean"),
            populist_share_of_group=("populist_share_of_group", "mean"),
        ).reset_index()
        grp["dimension"] = dim_name
        long_frames.append(grp)

    df_long = pd.concat(long_frames, ignore_index=True)
    df_long["election_year"] = df_long[year_col].isin(EP_ELECTION_YEARS).astype(int)
    df_long["crisis_year"] = df_long[year_col].isin(EP_CRISIS_YEARS).astype(int)


    if min_sentences is not None:
        n_before = df_long[df_long["dimension"] == "anti_elitism"].shape[0]
        thin = df_long["n_sentences"] < min_sentences
        df_long = df_long[~thin]
        n_after = df_long[df_long["dimension"] == "anti_elitism"].shape[0]
        print(f"Dropped {n_before - n_after} party-years with "
              f"< {min_sentences} sentences.")

    return df_long.rename(columns={year_col: "year", party_col: "party_group"})


IVS = ["election_year", "crisis_year", "populist_share_of_group", "populist_ep_share"]
DIMENSIONS = ["anti_elitism", "people_centrism", "populism"]


# ---------------------------------------------------------------------------
# 1. QUICK DIAGNOSTICS: multicollinearity check on the full model 
#       (VIF > 5 signals problematic collinearity)
# ---------------------------------------------------------------------------

def check_vif(df, ivs=IVS):
    X = sm.add_constant(df[ivs].dropna())
    vif = pd.DataFrame({
        "variable": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })
    return vif[vif["variable"] != "const"]

# ---------------------------------------------------------------------------
# 1b. SAVE A REGRESSION TABLE AS A PNG
# ---------------------------------------------------------------------------
 
def save_table_as_png(table, filepath, fontsize=9, dpi=200):
    text = str(table)
    lines = text.split("\n")
    char_width = max(len(line) for line in lines)
    fig_width = max(6, char_width * fontsize * 0.011)
    fig_height = max(2, len(lines) * fontsize * 0.022)
 
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.text(0, 1, text, family="monospace", fontsize=fontsize,
            va="top", ha="left", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filepath}")

# ---------------------------------------------------------------------------
# 1c. EXTRACT TIDY RESULTS (for plotting in a separate script)
# ---------------------------------------------------------------------------

def tidy_panelols(model, dimension, model_type="pooled_full", party=None):
    ci = model.conf_int()
    rows = []
    for term in model.params.index:
        rows.append({
            "dimension": dimension,
            "model_type": model_type,
            "party": party,          # None for pooled models
            "term": term,
            "coefficient": model.params[term],
            "std_error": model.std_errors[term],
            "ci_low": ci.loc[term, "lower"],
            "ci_high": ci.loc[term, "upper"],
            "p_value": model.pvalues[term],
            "n_obs": int(model.nobs),
            "r_squared": model.rsquared,
        })
    return rows

def tidy_ols(model, dimension, model_type="by_party", party=None):
    ci = model.conf_int()
    rows = []
    for term in model.params.index:
        rows.append({
            "dimension": dimension,
            "model_type": model_type,
            "party": party,
            "term": term,
            "coefficient": model.params[term],
            "std_error": model.bse[term],
            "ci_low": ci.loc[term, 0],
            "ci_high": ci.loc[term, 1],
            "p_value": model.pvalues[term],
            "n_obs": int(model.nobs),
            "r_squared": model.rsquared,
        })
    return rows

def tidy_mixedlm(mdf, dimension, model_type="mixed_effects"):
    rows = []
    ci = mdf.conf_int()
    for term in mdf.fe_params.index:
        rows.append({
            "dimension": dimension, "model_type": model_type, "party": None,
            "term": term, "coefficient": mdf.fe_params[term],
            "std_error": mdf.bse_fe[term],
            "ci_low": ci.loc[term, 0], "ci_high": ci.loc[term, 1],
            "p_value": mdf.pvalues[term], "n_obs": int(mdf.nobs),
            "r_squared": np.nan,
        })
    for party, effects in mdf.random_effects.items():
        for term, val in effects.items():
            rows.append({
                "dimension": dimension, "model_type": model_type + "_random_effect",
                "party": party, "term": term, "coefficient": val,
                "std_error": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "p_value": np.nan, "n_obs": int(mdf.nobs), "r_squared": np.nan,
            })
    return rows

def save_tidy_results(all_rows, filepath="../outputs/regression/regression_results_tidy.csv"):
    tidy_df = pd.DataFrame(all_rows)
    tidy_df.to_csv(filepath, index=False)
    print(f"Saved tidy results: {filepath} ({len(tidy_df)} rows)")
    return tidy_df
 

# ---------------------------------------------------------------------------
# 2. UNIVARIATE -> FULL MODEL PROGRESSION, for one dimension
# ---------------------------------------------------------------------------
#
# Uses linearmodels.PanelOLS with:
#   - party (entity) fixed effects  -> controls for stable party-level
#     differences in baseline rhetoric
#   - Driscoll-Kraay ("kernel") covariance -> robust to serial correlation
#     within party AND correlation across parties in the same year

def _prep_panel(df, entity_col="party_group", time_col="year"):
    return df.set_index([entity_col, time_col])

def run_model_progression(df, dv="rhetoric_pct", ivs=IVS,
                           entity_col="party_group", time_col="year",
                           entity_effects=True, weight_col="n_sentences"):
 
    panel = _prep_panel(df, entity_col, time_col)
    weights = panel[weight_col] if weight_col else None
    models = {}

    for iv in ivs:
        exog = sm.add_constant(panel[[iv]])
        m = PanelOLS(panel[dv], exog, entity_effects=entity_effects,
                      weights=weights).fit(cov_type="kernel")
        models[iv] = m

    exog_full = sm.add_constant(panel[ivs])
    m_full = PanelOLS(panel[dv], exog_full, entity_effects=entity_effects,
                       weights=weights).fit(cov_type="kernel")
    models["full"] = m_full

    table = compare(models, stars=True)
    return models, table


# ---------------------------------------------------------------------------
# 3. PER-PARTY-GROUP MODELS
# ---------------------------------------------------------------------------

def run_per_group(df, dv="rhetoric_pct", ivs=IVS, group_col="party_group",
                   time_col="year", maxlags=None):

    results = {}
    formula_full = f"{dv} ~ " + " + ".join(ivs)

    for grp, sub in df.groupby(group_col):
        if sub.shape[0] < len(ivs) + 2:
            print(f"Skipping {grp}: not enough observations ({sub.shape[0]})")
            continue
        sub = sub.sort_values(time_col)
        lags = maxlags or int(np.floor(4 * (sub.shape[0] / 100) ** (2 / 9))) or 1
        m = smf.ols(formula_full, data=sub).fit(
            cov_type="HAC", cov_kwds={"maxlags": lags}
        )
        results[grp] = m

    table = summary_col(
        list(results.values()),
        model_names=list(results.keys()),
        stars=True,
        info_dict={"N": lambda x: f"{int(x.nobs)}"}
    )
    return results, table


# ---------------------------------------------------------------------------
# WORKFLOW
# ---------------------------------------------------------------------------

import os
os.makedirs("../outputs/regression", exist_ok=True)

if __name__ == "__main__":

    df_sentences = pd.read_csv('../data/ep_speeches_populist_party_data.csv')

    # --- 1. Aggregate sentence-level data to party x year x dimension ---
    df = aggregate_to_party_year(df_sentences, min_sentences=10)
    df.to_csv("../outputs/regression/party_year_aggregated.csv", index=False)  # save intermediate

    df = pd.read_csv("../outputs/regression/party_year_aggregated.csv")

    # --- 1b. Collinearity check on full IV set (run on ONE dimension's rows) ---
    print(check_vif(df[df["dimension"] == "populism"]))

    # Collect csv for later visualisation
    tidy_rows = []

    # --- 2. Univariate -> full progression, per dimension ---
    for dim in DIMENSIONS:
        sub = df[df["dimension"] == dim]
        models, table = run_model_progression(sub)
        print(f"\n=== {dim.upper()}: pooled, all parties ===")
        print(table)
        save_table_as_png(table, f"../outputs/regression/{dim}_table.png")
        tidy_rows += tidy_panelols(models["full"], dimension=dim, model_type="pooled_full")

    # --- 3. Per-party models (full model only), per dimension ---
    for dim in DIMENSIONS:
        sub = df[df["dimension"] == dim]
        results, table = run_per_group(sub)
        print(f"\n=== {dim.upper()}: by party ===")
        print(table)
        save_table_as_png(table, f"../outputs/regression/{dim}_by_party_table.png")
        for party, m in results.items():
            tidy_rows += tidy_ols(m, dimension=dim, model_type="by_party", party=party)


    save_tidy_results(tidy_rows)