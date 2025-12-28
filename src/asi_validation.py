"""
ASI Validation Module

Functions for validating ASI metrics against positional expectations.
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def categorize_position(acronym: str) -> str:
    """
    Categorize player role acronym into broader position category.

    Args:
        acronym: Player role acronym (e.g., 'LM', 'CB', 'GK')

    Returns:
        Position category string
    """
    if acronym in ['GK']:
        return 'Goalkeeper'
    elif acronym in ['CB', 'LCB', 'RCB', 'LB', 'RB', 'LWB', 'RWB']:
        return 'Defender'
    elif acronym in ['DM', 'LDM', 'RDM', 'CM', 'LCM', 'RCM']:
        return 'Defensive/Central Mid'
    elif acronym in ['AM', 'LAM', 'RAM', 'LM', 'RM', 'LW', 'RW']:
        return 'Attacking Mid/Winger'
    elif acronym in ['CF', 'ST', 'LF', 'RF']:
        return 'Forward'
    return 'Other'


def calculate_position_stats(player_scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate ASI statistics by position category.

    Args:
        player_scores_df: DataFrame with player ASI scores (must have 'player_role_acronym' and 'asi_score')

    Returns:
        DataFrame with mean_asi, std_asi, num_players, total_opportunities per category
    """
    df = player_scores_df.copy()
    df['position_category'] = df['player_role_acronym'].apply(categorize_position)

    category_stats = df.groupby('position_category').agg({
        'asi_score': ['mean', 'std', 'count'],
        'opportunities': 'sum'
    }).round(3)
    category_stats.columns = ['mean_asi', 'std_asi', 'num_players', 'total_opportunities']
    category_stats = category_stats.sort_values('mean_asi', ascending=False)

    return category_stats


def plot_position_validation(category_stats: pd.DataFrame, show: bool = True) -> plt.Figure:
    """
    Create horizontal bar chart of ASI by position category.

    Args:
        category_stats: DataFrame from calculate_position_stats()
        show: Whether to display the plot

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ['#20c997', '#69db7c', '#ffd43b', '#ffa94d', '#ff6b6b']
    positions = category_stats.index.tolist()
    asi_values = category_stats['mean_asi'].values * 100

    bars = ax.barh(positions, asi_values, color=colors[:len(positions)], edgecolor='white', linewidth=2)

    ax.set_xlabel('Mean ASI Score (%)', fontsize=12)
    ax.set_title('ASI by Position Category (All 10 Matches)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)

    # Add value labels
    for bar, (idx, row) in zip(bars, category_stats.iterrows()):
        width = bar.get_width()
        ax.annotate(f'{width:.1f}% (n={int(row["num_players"])})',
                    xy=(width + 1, bar.get_y() + bar.get_height() / 2),
                    ha='left', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def test_position_significance(player_scores_df: pd.DataFrame) -> dict:
    """
    Perform Mann-Whitney U test comparing midfielders vs defenders.

    Args:
        player_scores_df: DataFrame with player ASI scores

    Returns:
        dict with test results: midfielders_mean, defenders_mean, n_mid, n_def, u_stat, p_value, significant
    """
    df = player_scores_df.copy()
    df['position_category'] = df['player_role_acronym'].apply(categorize_position)

    midfielders = df[df['position_category'].isin(['Attacking Mid/Winger', 'Defensive/Central Mid'])]['asi_score']
    defenders = df[df['position_category'] == 'Defender']['asi_score']

    stat, p_value = stats.mannwhitneyu(midfielders, defenders, alternative='greater')

    return {
        'midfielders_mean': midfielders.mean(),
        'defenders_mean': defenders.mean(),
        'n_midfielders': len(midfielders),
        'n_defenders': len(defenders),
        'u_statistic': stat,
        'p_value': p_value,
        'significant': p_value < 0.001
    }


def print_significance_result(result: dict) -> None:
    """Print formatted significance test result."""
    print(f"Midfielders vs Defenders:")
    print(f"  Midfielders mean ASI: {result['midfielders_mean']:.1%} (n={result['n_midfielders']})")
    print(f"  Defenders mean ASI: {result['defenders_mean']:.1%} (n={result['n_defenders']})")
    sig_marker = 'Significant (p < 0.001)' if result['significant'] else ''
    print(f"  Mann-Whitney U: {result['u_statistic']:.0f}, p = {result['p_value']:.2e} {sig_marker}")
