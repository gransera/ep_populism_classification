"""
Populist Rhetoric Visualizations
==================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ---------------------------------------------------------------------------
# LOAD TIDY RESULTS
# ---------------------------------------------------------------------------

tidy = pd.read_csv('../outputs/regression/regression_results_tidy.csv')

DIMENSIONS = ['anti_elitism', 'people_centrism', 'populism']

IVS_BINARY = ['election_year', 'crisis_year']
IVS_SHARES = ['populist_share_of_group', 'populist_ep_share']
IVS = IVS_BINARY + IVS_SHARES

iv_labels = {
    'election_year': 'Election Year',
    'crisis_year': 'Crisis Year',
    'populist_share_of_group': 'Populist Share\nof EP Group',
    'populist_ep_share': 'Populist Share\nof EP',
}

dim_labels = {
    'anti_elitism': 'Anti-Elitism',
    'people_centrism': 'People-Centrism',
    'populism': 'Populism (Combined)',
}

dim_colors = {
    'anti_elitism': '#2166ac',
    'people_centrism': '#d6604d',
    'populism': '#4dac26',
}

# ---------------------------------------------------------------------------
# 1. COEFFICIENT PLOT — pooled full model, all three dimensions
# ---------------------------------------------------------------------------

def plot_coefficient_forest(tidy, ivs, title, filename, model_type='pooled_full'):
    df = tidy[(tidy['model_type'] == model_type) & (tidy['term'].isin(ivs))]

    fig, ax = plt.subplots(figsize=(10, 5))
    n_dims = len(DIMENSIONS)
    offsets = np.linspace(-0.2, 0.2, n_dims)

    for i, (dim, offset) in enumerate(zip(DIMENSIONS, offsets)):
        sub = df[df['dimension'] == dim].set_index('term')
        y_positions = np.arange(len(ivs)) + offset

        ax.errorbar(
            x=sub.loc[ivs, 'coefficient'],
            y=y_positions,
            xerr=[
                sub.loc[ivs, 'coefficient'] - sub.loc[ivs, 'ci_low'],
                sub.loc[ivs, 'ci_high'] - sub.loc[ivs, 'coefficient']
            ],
            fmt='o', color=dim_colors[dim], label=dim_labels[dim],
            capsize=4, linewidth=1.5, markersize=6
        )

    ax.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_yticks(np.arange(len(ivs)))
    ax.set_yticklabels([iv_labels[iv] for iv in ivs], fontsize=11)
    ax.set_xlabel('Coefficient (% of Sentences)', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(title='Dimension', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'../outputs/regression/{filename}', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


# ---------------------------------------------------------------------------
# 2. PER-PARTY COEFFICIENT PLOT — one panel per IV
# ---------------------------------------------------------------------------

def plot_per_party_coefficients(tidy):
    df = tidy[(tidy['model_type'] == 'by_party') & (tidy['term'].isin(IVS))]
    parties = sorted(df['party'].dropna().unique())

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for ax, iv in zip(axes.flatten(), IVS):
        sub = df[df['term'] == iv].copy()

        for dim, color in dim_colors.items():
            d = sub[sub['dimension'] == dim].set_index('party').reindex(parties)
            y = np.arange(len(parties))
            offset = list(dim_colors.keys()).index(dim) * 0.2 - 0.2

            ax.errorbar(
                x=d['coefficient'],
                y=y + offset,
                xerr=[
                    d['coefficient'] - d['ci_low'],
                    d['ci_high'] - d['coefficient']
                ],
                fmt='o', color=color, label=dim_labels[dim],
                capsize=3, linewidth=1.2, markersize=5
            )

        ax.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.set_yticks(np.arange(len(parties)))
        ax.set_yticklabels(parties, fontsize=9)
        ax.set_title(iv_labels[iv], fontsize=12, fontweight='bold')
        ax.set_xlabel('Coefficient', fontsize=10)
        ax.grid(axis='x', alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Dimension', loc='lower center',
               ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Per-Party Coefficients by IV and Dimension', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../outputs/regression/viz_per_party_coefficients.png', dpi=200, bbox_inches='tight')
    print("Saved: viz_per_party_coefficients.png")


# ---------------------------------------------------------------------------
# 3a. R² BAR CHART — POOLED MODEL ONLY (valid cross-dimension comparison)
# ---------------------------------------------------------------------------
#
# Deliberately NOT shown alongside per-party R² -- pooled R² is R²-within
# (variance left after removing each party's own average level via entity
# fixed effects), while per-party R² is unconditional/total R² for a
# single time series with no such baseline removed. They answer different
# questions and are not comparable on the same axis; showing them side by
# side (even using the median) previously implied a false comparison.

def plot_r2_pooled(tidy):
    pooled = (tidy[tidy['model_type'] == 'pooled_full']
              .groupby('dimension')['r_squared'].first()
              .reindex(DIMENSIONS))

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(range(len(DIMENSIONS)), pooled.values,
                   color=[dim_colors[d] for d in DIMENSIONS], alpha=0.85)
    ax.bar_label(bars, fmt='%.2f', fontsize=10, padding=3)

    ax.set_xticks(range(len(DIMENSIONS)))
    ax.set_xticklabels([dim_labels[d] for d in DIMENSIONS], fontsize=11)
    ax.set_ylabel('R² (within)', fontsize=11)
    ax.set_title('Pooled Model Fit (R²) by Dimension', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('../outputs/regression/viz_r2_pooled.png', dpi=200, bbox_inches='tight')
    print("Saved: viz_r2_pooled.png")


# ---------------------------------------------------------------------------
# 3b. R² SPREAD ACROSS PARTIES — for the heterogeneity/robustness section
# ---------------------------------------------------------------------------
#
# Shows how much per-party R² varies across the 11 parties, per dimension
# -- a legitimate heterogeneity statistic on its own. Deliberately kept
# separate from the pooled chart above (see note in figure caption) and
# individual party R² values are overlaid as points so small-N parties
# (which can show inflated R² from overfitting) are visible rather than
# hidden inside a summary statistic.

def plot_r2_per_party_spread(tidy):
    by_party = (tidy[tidy['model_type'] == 'by_party']
                .groupby(['dimension', 'party'])
                .first()[['r_squared', 'n_obs']].reset_index())

    fig, ax = plt.subplots(figsize=(8, 5.5))
    data = [by_party[by_party['dimension'] == d]['r_squared'].values for d in DIMENSIONS]

    try:
        bp = ax.boxplot(data, tick_labels=[dim_labels[d] for d in DIMENSIONS],
                         patch_artist=True, widths=0.5, showfliers=False)
    except TypeError:  # matplotlib < 3.9 doesn't have tick_labels yet
        bp = ax.boxplot(data, labels=[dim_labels[d] for d in DIMENSIONS],
                         patch_artist=True, widths=0.5, showfliers=False)
    for patch, d in zip(bp['boxes'], DIMENSIONS):
        patch.set_facecolor(dim_colors[d])
        patch.set_alpha(0.45)

    rng = np.random.default_rng(0)
    for i, d in enumerate(DIMENSIONS):
        sub = by_party[by_party['dimension'] == d]
        x_jitter = rng.normal(i + 1, 0.05, size=len(sub))
        # smaller N -> more likely inflated fit: mark N<=15 parties distinctly
        small_n = sub['n_obs'] <= 15
        ax.scatter(x_jitter[~small_n.values], sub.loc[~small_n, 'r_squared'],
                   color='black', alpha=0.7, s=25, zorder=3, label='N > 15' if i == 0 else None)
        ax.scatter(x_jitter[small_n.values], sub.loc[small_n, 'r_squared'],
                   color='black', alpha=0.7, s=45, marker='^', zorder=3,
                   label='N \u2264 15 (caution)' if i == 0 else None)

    ax.set_ylabel('R² (individual per-party model)', fontsize=11)
    ax.set_title('Spread of Per-Party R² by Dimension', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)

    note = ("Note: per-party R\u00b2 is not comparable to pooled R\u00b2-within (different\n"
            "baseline; small-N parties, marked \u25b2, can show inflated fit from overfitting).")
    fig.text(0.5, -0.05, note, ha='center', fontsize=8, style='italic')

    plt.tight_layout()
    plt.savefig('../outputs/regression/viz_r2_per_party_spread.png', dpi=200, bbox_inches='tight')
    print("Saved: viz_r2_per_party_spread.png")


# ---------------------------------------------------------------------------
# 4. SIGNIFICANCE HEATMAP — p-values per IV x dimension (pooled model)
# ---------------------------------------------------------------------------

def plot_significance_heatmap(tidy):
    df = tidy[(tidy['model_type'] == 'pooled_full') & (tidy['term'].isin(IVS))]
    pivot = df.pivot(index='term', columns='dimension', values='p_value').reindex(IVS)[DIMENSIONS]

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(pivot.values, cmap='RdYlGn_r', vmin=0, vmax=0.1, aspect='auto')

    ax.set_xticks(range(len(DIMENSIONS)))
    ax.set_xticklabels([dim_labels[d] for d in DIMENSIONS], fontsize=11)
    ax.set_yticks(range(len(IVS)))
    ax.set_yticklabels([iv_labels[iv] for iv in IVS], fontsize=11)

    for i in range(len(IVS)):
        for j in range(len(DIMENSIONS)):
            val = pivot.values[i, j]
            stars = '***' if val < 0.01 else '**' if val < 0.05 else '*' if val < 0.1 else ''
            ax.text(j, i, f'{val:.3f}{stars}', ha='center', va='center', fontsize=9)

    plt.colorbar(im, ax=ax, label='p-value')
    ax.set_title('P-value Heatmap (Pooled Full Model)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../outputs/regression/viz_pvalue_heatmap.png', dpi=200, bbox_inches='tight')
    print("Saved: viz_pvalue_heatmap.png")

# ---------------------------------------------------------------------------
# 5. SEAT SHARE VARIABLES ACROSS PARTY GROUPS
# ---------------------------------------------------------------------------

def plot_seat_shares_by_group(tidy):
    df = tidy[
        (tidy['model_type'] == 'by_party') &
        (tidy['term'].isin(['populist_share_of_group', 'populist_ep_share']))
    ].copy()

    for iv in ['populist_share_of_group', 'populist_ep_share']:
        sub = df[df['term'] == iv]
        parties = sorted(sub['party'].dropna().unique())

        fig, ax = plt.subplots(figsize=(10, 6))
        n_dims = len(DIMENSIONS)
        offsets = np.linspace(-0.2, 0.2, n_dims)

        for dim, offset in zip(DIMENSIONS, offsets):
            d = sub[sub['dimension'] == dim].set_index('party').reindex(parties)
            y = np.arange(len(parties)) + offset

            ax.errorbar(
                x=d['coefficient'],
                y=y,
                xerr=[
                    d['coefficient'] - d['ci_low'],
                    d['ci_high'] - d['coefficient']
                ],
                fmt='o', color=dim_colors[dim], label=dim_labels[dim],
                capsize=3, linewidth=1.2, markersize=5
            )

        ax.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.set_yticks(np.arange(len(parties)))
        ax.set_yticklabels(parties, fontsize=10)
        ax.set_xlabel('Coefficient (% of Sentences)', fontsize=11)
        ax.set_title(f'{iv_labels[iv]}: Per-Party Coefficients by Dimension',
                     fontsize=13, fontweight='bold')
        ax.legend(title='Dimension', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        fname = f'viz_{iv}_by_party.png'
        plt.savefig(f'../outputs/regression/{fname}', dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved: {fname}")


def plot_seat_shares_combined(tidy):
    df = tidy[
        (tidy['model_type'] == 'by_party') &
        (tidy['term'].isin(['populist_share_of_group', 'populist_ep_share'])) &
        (tidy['dimension'] == 'populism')
    ].copy()

    parties = sorted(df['party'].dropna().unique())
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, iv in zip(axes, ['populist_share_of_group', 'populist_ep_share']):
        sub = df[df['term'] == iv].set_index('party').reindex(parties)

        ax.errorbar(
            x=sub['coefficient'],
            y=np.arange(len(parties)),
            xerr=[
                sub['coefficient'] - sub['ci_low'],
                sub['ci_high'] - sub['coefficient']
            ],
            fmt='o', color='#4dac26', capsize=4, linewidth=1.5, markersize=6
        )

        ax.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.set_yticks(np.arange(len(parties)))
        ax.set_yticklabels(parties, fontsize=10)
        ax.set_xlabel('Coefficient (% of Sentences)', fontsize=11)
        ax.set_title(iv_labels[iv], fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

    fig.suptitle('Seat Share Effects by Party Group (Populism Dimension)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../outputs/regression/viz_seat_shares_populism.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: viz_seat_shares_populism.png")


# ---------------------------------------------------------------------------
# RUN ALL
# ---------------------------------------------------------------------------

#plot_coefficient_forest(tidy, IVS_BINARY, 
#                        'Pooled Panel Model: Binary Variables', 
#                        'viz_coefficient_forest_binary.png')

#plot_coefficient_forest(tidy, IVS_SHARES,
#                        'Pooled Panel Model: Seat Share Variables',
#                        'viz_coefficient_forest_shares.png')

#plot_per_party_coefficients(tidy)
#plot_r2_pooled(tidy)
#plot_r2_per_party_spread(tidy)
#plot_significance_heatmap(tidy)
plot_seat_shares_by_group(tidy)
plot_seat_shares_combined(tidy)