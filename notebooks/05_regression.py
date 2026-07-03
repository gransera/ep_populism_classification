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

EP_ELECTION_YEARS = [1999, 2004, 2009, 2014, 2019, 2024]
EP_CRISIS_YEARS = [2004, 2009, 2010, 2015, 2016, 2020, 2021, 2022, 2023] # Eastern enlargement round, Euro area crisis, Refugee crisis, Covid-19, Energy crisis (Russian war)


# ---------------------------------------------------------------------------
# 1. AGGREGATE SENTENCE-LEVEL DATA -> PARTY x YEAR x DIMENSION
# ---------------------------------------------------------------------------

def aggregate_to_party_year(df_sentences,
                             party_col="party",
                             year_col="legislative_year",
                             dim_binary_cols=None,
                             min_sentences=None):
    """
    Collapses sentence-level data to one row per (party, year, dimension).

    dim_binary_cols : dict mapping output dimension name -> sentence-level
        binary column, defaults to your actual columns:
            {"anti_elitism":    "anti_elitism_binary",
             "people_centrism": "people_centrism_binary",
             "populism":        "populism_combined"}

    Produces rhetoric_pct = 100 * mean(binary flag) within each party-year,
    plus n_sentences (needed for weighting -- see run_model_progression's
    `weight_col` argument) and the seat-share IVs, election_year.

    min_sentences : optionally drop party-years with fewer than this many
        sentences (very thin party-years give unstable percentages).
    """
    if dim_binary_cols is None:
        dim_binary_cols = {
            "anti_elitism": "anti_elitism_binary",
            "people_centrism": "people_centrism_binary",
            "populism": "populism_combined",
        }

    # --- sanity check: seat-share columns should be constant within a
    # given party-year (same value for every sentence from that party in
    # that year). Flag if not -- would indicate a data issue or that these
    # vary at a finer grain than expected (e.g. by speech date within year).
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


def check_populism_combined_construction(df_sentences,
                                          anti="anti_elitism_binary",
                                          people="people_centrism_binary",
                                          combined="populism_combined"):
    """
    Diagnostic: is populism_combined mechanically derived from the other
    two flags (e.g. AND / OR), or independently coded? This matters for
    how you interpret "three comparable dimensions" -- if populism is a
    logical combination of the other two, its regression results will be
    mechanically related to theirs, not an independent third measure.
    Run this once and eyeball the crosstabs.
    """
    df = df_sentences
    and_match = ((df[anti] & df[people]) == df[combined]).mean()
    or_match = ((df[anti] | df[people]) == df[combined]).mean()
    print(f"Share matching AND(anti_elitism, people_centrism): {and_match:.3f}")
    print(f"Share matching OR(anti_elitism, people_centrism):  {or_match:.3f}")
    print("\nCrosstab (anti_elitism x people_centrism -> populism_combined mean):")
    print(df.groupby([anti, people])[combined].mean())


# IVS uses your actual seat-share column names (post-aggregation).
# crisis_year deliberately omitted for now -- add it to this list once
# you've defined it, and make sure aggregate_to_party_year() creates the
# corresponding column too.
IVS = ["election_year", "crisis_year", "populist_share_of_group", "populist_ep_share"]
DIMENSIONS = ["anti_elitism", "people_centrism", "populism"]


# ---------------------------------------------------------------------------
# 1. QUICK DIAGNOSTICS: multicollinearity check on the full model
# ---------------------------------------------------------------------------

def check_vif(df, ivs=IVS):
    """Run this once on the full IV set before trusting the 'all together' model."""
    X = sm.add_constant(df[ivs].dropna())
    vif = pd.DataFrame({
        "variable": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })
    return vif[vif["variable"] != "const"]

# Rule of thumb: VIF > 5 (some say >10) signals problematic collinearity.
# pop_seat_share_group and pop_seat_share_ep are the most likely pair to
# flag here since a group's seat share partly drives the EP-wide figure.

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
# 2. UNIVARIATE -> FULL MODEL PROGRESSION, for one dimension
# ---------------------------------------------------------------------------
#
# Uses linearmodels.PanelOLS with:
#   - party (entity) fixed effects  -> controls for stable party-level
#     differences in baseline rhetoric
#   - Driscoll-Kraay ("kernel") covariance -> robust to serial correlation
#     within party AND correlation across parties in the same year
#
# NOTE: entity_effects=True, but deliberately NO time_effects=True, since
# election_year / crisis_year vary only over time and would be wiped out
# by year fixed effects.

def _prep_panel(df, entity_col="party_group", time_col="year"):
    """Set the (entity, time) MultiIndex PanelOLS expects."""
    return df.set_index([entity_col, time_col])

def run_model_progression(df, dv="rhetoric_pct", ivs=IVS,
                           entity_col="party_group", time_col="year",
                           entity_effects=True, weight_col="n_sentences"):
    """
    Runs: DV ~ IV1, DV ~ IV2, ..., DV ~ IV1+IV2+IV3+IV4
    Party fixed effects + Driscoll-Kraay SEs (bandwidth default ~ T^(1/4)).
    Returns a dict of fitted models plus a linearmodels comparison table.

    weight_col : if set (default "n_sentences", produced by
        aggregate_to_party_year), runs WEIGHTED least squares using
        sentence count as weight. This matters because rhetoric_pct is a
        proportion -- a party-year built from 400 sentences is a much
        more precise estimate than one built from 8, and unweighted OLS
        treats them as equally informative. Set to None to disable.
    """
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
    """
    Full model (all IVs together) run separately within each party group.
    Each party group here is a genuine 26-year time series (1999-2024), so
    autocorrelation is a real concern -> HAC (Newey-West) SEs, not plain
    OLS or simple HC robust SEs. Default maxlags follows the common
    rule-of-thumb floor(4*(T/100)^(2/9)); override if you have a reason to.
    """
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
# 4a. THREE DIMENSIONS SIDE BY SIDE (Option A - recommended main output)
# ---------------------------------------------------------------------------

def run_all_dimensions(df_long, ivs=IVS, entity_col="party_group",
                        time_col="year", dim_col="dimension",
                        dv="rhetoric_pct", dimensions=DIMENSIONS,
                        entity_effects=True):
    """
    Runs the same full panel model (party FE + Driscoll-Kraay SEs)
    separately for each dimension and puts them in one comparison table
    (columns = anti_elitism / people_centrism / populism).
    """
    models = {}
    for dim in dimensions:
        sub = df_long[df_long[dim_col] == dim]
        panel = _prep_panel(sub, entity_col, time_col)
        exog = sm.add_constant(panel[ivs])
        m = PanelOLS(panel[dv], exog, entity_effects=entity_effects).fit(
            cov_type="kernel"
        )
        models[dim] = m

    table = compare(models, stars=True)
    return models, table


# ---------------------------------------------------------------------------
# 4b. SEEMINGLY UNRELATED REGRESSION across the three dimensions (Option B)
# ---------------------------------------------------------------------------

def run_sur(df_wide, ivs=IVS, dv_cols=None):
    """
    Joint estimation of the three dimension-equations, exploiting
    correlation between their residuals. Requires WIDE data: one row per
    party_group x year, with the three DV columns.

    dv_cols : dict, e.g. {"anti_elitism": "anti_elitism_pct",
                           "people_centrism": "people_centrism_pct",
                           "populism": "populism_pct"}
    """
    from linearmodels.system import SUR

    if dv_cols is None:
        dv_cols = {d: f"{d}_pct" for d in DIMENSIONS}

    equations = {}
    for dim, col in dv_cols.items():
        exog = sm.add_constant(df_wide[ivs])
        equations[dim] = {"dependent": df_wide[col], "exog": exog}

    sur_model = SUR(equations)
    sur_res = sur_model.fit(cov_type="robust")
    return sur_res


# ---------------------------------------------------------------------------
# EXAMPLE WORKFLOW
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    df_sentences = pd.read_csv('../data/ep_speeches_populist_party_data.csv')

    # --- 0. Diagnose how populism_combined relates to the other two flags ---
    check_populism_combined_construction(df_sentences)

    # --- 1. Aggregate sentence-level data to party x year x dimension ---
    df = aggregate_to_party_year(df_sentences, min_sentences=10)
    df.to_csv("../outputs/party_year_aggregated.csv", index=False)  # save intermediate

    # --- 2. Collinearity check on full IV set (run on ONE dimension's rows) ---
    print(check_vif(df[df["dimension"] == "populism"]))

    # --- 3. Univariate -> full progression, per dimension ---
    for dim in DIMENSIONS:
        sub = df[df["dimension"] == dim]
        models, table = run_model_progression(sub)
        print(f"\n=== {dim.upper()}: pooled, all parties ===")
        print(table)
        save_table_as_png(table, f"../outputs/{dim}_table.png")

    # --- 4. Per-party models (full model only), per dimension ---
    for dim in DIMENSIONS:
        sub = df[df["dimension"] == dim]
        results, table = run_per_group(sub)
        print(f"\n=== {dim.upper()}: by party ===")
        print(table)
        save_table_as_png(table, f"../outputs/{dim}_by_party_table.png")


    # --- 5a. Three dimensions side by side ---
    models, table = run_all_dimensions(df)
    print("\n=== Full model, all three dimensions compared ===")
    print(table)
    save_table_as_png(table, "../outputs/all_dimensions_comparison_table.png")

    # --- 5b. SUR (needs wide-format: one row per party-year, 3 DV columns) ---
    # df_wide = df.pivot_table(index=["party_group", "year"] + IVS,
    #                           columns="dimension", values="rhetoric_pct").reset_index()
    # df_wide.columns.name = None
    # sur_res = run_sur(df_wide, dv_cols={d: d for d in DIMENSIONS})
    # print(sur_res)
