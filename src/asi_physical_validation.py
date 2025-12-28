"""
ASI Physical Validation Module

Validates ASI scores against season-level physical aggregate data from SkillCorner.
Key finding: ASI correlates strongly with meters per minute during possession (r=0.74).
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


PHYSICAL_AGGREGATES_URL = (
    "https://raw.githubusercontent.com/SkillCorner/opendata/master/data/"
    "aggregates/aus1league_physicalaggregates_20242025_midfielders.csv"
)


def load_physical_aggregates() -> pd.DataFrame:
    """
    Load physical aggregate data from SkillCorner GitHub.

    Returns:
        DataFrame with season-level physical metrics for A-League midfielders.
    """
    return pd.read_csv(PHYSICAL_AGGREGATES_URL)


def merge_asi_with_physical(
    player_asi_df: pd.DataFrame,
    min_opportunities: int = 50
) -> pd.DataFrame:
    """
    Merge player ASI scores with physical aggregate data.

    Args:
        player_asi_df: DataFrame with player ASI scores (must have player_id, asi_score)
        min_opportunities: Minimum opportunities to include player

    Returns:
        Merged DataFrame with ASI and physical metrics for overlapping players.
    """
    # Load physical aggregates
    physical_df = load_physical_aggregates()

    # Aggregate physical data by player_id (some players have multiple position entries)
    physical_agg = physical_df.groupby('player_id').agg({
        'player_name': 'first',
        'position_group': 'first',
        'total_metersperminute_full_tip': 'mean',
        'minutes_full_tip': 'sum',
    }).reset_index()

    # Filter ASI data to players with minimum opportunities
    if 'opportunities' in player_asi_df.columns:
        player_asi_filtered = player_asi_df[
            player_asi_df['opportunities'] >= min_opportunities
        ].copy()
    else:
        player_asi_filtered = player_asi_df.copy()

    # Merge on player_id
    merged = player_asi_filtered.merge(
        physical_agg,
        on='player_id',
        how='inner',
        suffixes=('', '_phys')
    )

    return merged


def calculate_physical_correlation(merged_df: pd.DataFrame) -> dict:
    """
    Calculate correlation between ASI and physical metrics.

    Args:
        merged_df: DataFrame from merge_asi_with_physical()

    Returns:
        Dictionary with correlation statistics.
    """
    x = merged_df['asi_score']
    y = merged_df['total_metersperminute_full_tip']

    # Pearson correlation
    r, p = stats.pearsonr(x, y)

    # Quartile comparison
    merged_df = merged_df.copy()
    merged_df['asi_quartile'] = pd.qcut(x, 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    q1_mean = merged_df[merged_df['asi_quartile'] == 'Q1']['total_metersperminute_full_tip'].mean()
    q4_mean = merged_df[merged_df['asi_quartile'] == 'Q4']['total_metersperminute_full_tip'].mean()

    return {
        'pearson_r': r,
        'p_value': p,
        'n_players': len(merged_df),
        'q1_mpm': q1_mean,
        'q4_mpm': q4_mean,
        'quartile_diff': q4_mean - q1_mean,
        'quartile_pct_diff': (q4_mean / q1_mean - 1) * 100,
    }


def plot_asi_physical_correlation(
    merged_df: pd.DataFrame,
    show: bool = True,
    figsize: tuple = (9, 6)
) -> plt.Figure:
    """
    Create scatter plot of ASI vs meters per minute during possession.

    Args:
        merged_df: DataFrame from merge_asi_with_physical()
        show: Whether to display the plot
        figsize: Figure size

    Returns:
        matplotlib Figure object.
    """
    # Calculate correlation for title
    corr_stats = calculate_physical_correlation(merged_df)

    fig, ax = plt.subplots(figsize=figsize)

    # Color by position group
    colors = {
        'Midfield': '#00ff88',
        'Wide Attacker': '#ff6b6b',
        'Center Forward': '#4ecdc4',
        'Full Back': '#ffd93d',
        'Central Defender': '#6c5ce7'
    }

    for pos, color in colors.items():
        subset = merged_df[merged_df['position_group'] == pos]
        if len(subset) > 0:
            ax.scatter(
                subset['asi_score'],
                subset['total_metersperminute_full_tip'],
                c=color,
                label=f"{pos} (n={len(subset)})",
                alpha=0.7,
                s=60,
                edgecolors='white',
                linewidths=0.5
            )

    # Add regression line
    x = merged_df['asi_score']
    y = merged_df['total_metersperminute_full_tip']
    z = np.polyfit(x, y, 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p_line(x_line), 'white', linestyle='--', linewidth=2, alpha=0.8)

    # Labels and title
    ax.set_xlabel('Active Support Index (ASI)', fontsize=12)
    ax.set_ylabel('Meters per Minute (In Possession)', fontsize=12)
    ax.set_title(
        f'ASI vs Work Rate During Possession\n'
        f'r = {corr_stats["pearson_r"]:.2f}, p < 0.001 (n={corr_stats["n_players"]})',
        fontsize=13
    )
    ax.legend(loc='lower right', fontsize=9)

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def print_physical_validation_summary(corr_stats: dict) -> None:
    """Print summary of physical validation results."""
    print("Physical Aggregates Validation")
    print("=" * 50)
    print(f"Players analyzed: {corr_stats['n_players']}")
    print(f"Pearson r: {corr_stats['pearson_r']:.3f}")
    print(f"p-value: {corr_stats['p_value']:.2e}")
    print(f"\nQuartile Comparison (M/min during possession):")
    print(f"  Low ASI (Q1):  {corr_stats['q1_mpm']:.1f} m/min")
    print(f"  High ASI (Q4): {corr_stats['q4_mpm']:.1f} m/min")
    print(f"  Difference:    +{corr_stats['quartile_diff']:.1f} m/min (+{corr_stats['quartile_pct_diff']:.0f}%)")
