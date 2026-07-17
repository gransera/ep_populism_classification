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
# 3. R² BAR CHART — pooled vs per-party, per dimension
# ---------------------------------------------------------------------------

def plot_r2(tidy):
    pooled = (tidy[tidy['model_type'] == 'pooled_full']
              .groupby('dimension')['r_squared'].first().reset_index())
    pooled['model'] = 'Pooled'

    by_party = (tidy[tidy['model_type'] == 'by_party']
                .groupby(['dimension', 'party'])['r_squared'].first().reset_index()
                .groupby('dimension')['r_squared'].median().reset_index())
    by_party['model'] = 'Per-Party (median)'

    df_r2 = pd.concat([pooled, by_party])

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(DIMENSIONS))
    width = 0.35

    for i, (model, color) in enumerate([('Pooled', '#2166ac'), ('Per-Party (median)', '#d6604d')]):
        vals = df_r2[df_r2['model'] == model].set_index('dimension').reindex(DIMENSIONS)['r_squared']
        bars = ax.bar(x + i * width, vals, width, label=model, color=color, alpha=0.85)
        ax.bar_label(bars, fmt='%.2f', fontsize=9, padding=3)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([dim_labels[d] for d in DIMENSIONS], fontsize=11)
    ax.set_ylabel('R²', fontsize=11)
    ax.set_title('Model Fit (R²) by Dimension and Model Type', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('../outputs/regression/viz_r2_comparison.png', dpi=200, bbox_inches='tight')
    print("Saved: viz_r2_comparison.png")


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
# RUN ALL
# ---------------------------------------------------------------------------

plot_coefficient_forest(tidy, IVS_BINARY, 
                        'Pooled Panel Model: Binary Variables', 
                        'viz_coefficient_forest_binary.png')

plot_coefficient_forest(tidy, IVS_SHARES,
                        'Pooled Panel Model: Seat Share Variables',
                        'viz_coefficient_forest_shares.png')

plot_per_party_coefficients(tidy)
plot_r2(tidy)
plot_significance_heatmap(tidy)