"""
ASI Time Analysis Module

Functions for analyzing ASI patterns over match time and player fatigue.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from .asi_core import ASICalculator, ASIConfig
from .asi_data_loader import ASIDataLoader


def get_time_bin(match_time_min: float) -> str:
    """
    Bin match time into 15-minute intervals.

    Args:
        match_time_min: Match time in minutes

    Returns:
        Time bin label (e.g., '0-15', '15-30', etc.)
    """
    if match_time_min < 15:
        return '0-15'
    elif match_time_min < 30:
        return '15-30'
    elif match_time_min < 45:
        return '30-45'
    elif match_time_min < 60:
        return '45-60'
    elif match_time_min < 75:
        return '60-75'
    elif match_time_min < 90:
        return '75-90'
    else:
        return '90+'


def calculate_time_based_asi(loader: ASIDataLoader, config: ASIConfig,
                              match_ids: list = None) -> pd.DataFrame:
    """
    Calculate ASI metrics by time bin across multiple matches.

    Args:
        loader: ASIDataLoader instance
        config: ASIConfig instance
        match_ids: List of match IDs to process (default: all 10)

    Returns:
        DataFrame with all pressure events including time bin and half
    """
    if match_ids is None:
        match_ids = [1886347, 1899585, 1925299, 1953632, 1996435,
                     2006229, 2011166, 2013725, 2015213, 2017461]

    all_results = []

    for match_id in match_ids:
        try:
            match = loader.load_match(match_id)
            calc = ASICalculator(match, config=config)
            res = calc.process_all_pressure_events()

            # Get frame-to-time mapping from tracking data
            frame_time_map = match.tracking_df[['frame', 'match_time_min']].drop_duplicates().set_index('frame')['match_time_min']

            # Add match time to results
            res['match_time_min'] = res['frame_start'].map(frame_time_map)
            res['match_id'] = match_id
            all_results.append(res)
        except Exception as e:
            print(f"Match {match_id}: {e}")

    results_df = pd.concat(all_results, ignore_index=True)
    results_df['time_bin'] = results_df['match_time_min'].apply(get_time_bin)
    results_df['half'] = results_df['period'].apply(lambda x: 'First Half' if x == 1 else 'Second Half')

    return results_df


def get_time_bin_stats(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate ASI statistics by time bin.

    Args:
        results_df: DataFrame from calculate_time_based_asi()

    Returns:
        DataFrame with avg_support_ratio, avg_active_supporters, num_events per bin
    """
    time_asi = results_df.groupby('time_bin').agg({
        'active_support_ratio': 'mean',
        'num_active_supporters': 'mean',
        'event_id': 'count'
    }).round(3)
    time_asi.columns = ['avg_support_ratio', 'avg_active_supporters', 'num_events']

    # Reorder bins
    bin_order = ['0-15', '15-30', '30-45', '45-60', '60-75', '75-90', '90+']
    time_asi = time_asi.reindex([b for b in bin_order if b in time_asi.index])

    return time_asi


def plot_time_trend(time_asi: pd.DataFrame, show: bool = True) -> plt.Figure:
    """
    Create bar chart with trend line showing ASI over match time.

    Args:
        time_asi: DataFrame from get_time_bin_stats()
        show: Whether to display the plot

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    x_pos = range(len(time_asi))
    bars = ax.bar(x_pos, time_asi['avg_support_ratio'] * 100,
                  color=['#4dabf7' if i < 3 else '#f06595' for i in range(len(time_asi))],
                  edgecolor='white', linewidth=2, alpha=0.8)

    # Add trend line
    z = np.polyfit(list(x_pos), time_asi['avg_support_ratio'].values * 100, 1)
    p = np.poly1d(z)
    ax.plot(x_pos, p(list(x_pos)), 'w--', linewidth=2, alpha=0.8,
            label=f'Trend (slope: {z[0]:.2f}%/period)')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(time_asi.index, fontsize=11)
    ax.set_ylabel('Avg Active Support Ratio (%)', fontsize=12)
    ax.set_xlabel('Match Time (minutes)', fontsize=12)
    ax.set_title('Active Support Ratio Over Match Time (All 10 Matches)', fontsize=14, fontweight='bold')
    ax.axvline(x=2.5, color='white', linestyle=':', alpha=0.5, label='Halftime')
    ax.legend(loc='lower left')

    # Add value labels
    for bar, (idx, row) in zip(bars, time_asi.iterrows()):
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height + 0.5),
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def compare_halves(results_df: pd.DataFrame) -> dict:
    """
    Compare first half vs second half ASI with statistical test.

    Args:
        results_df: DataFrame from calculate_time_based_asi()

    Returns:
        dict with comparison results
    """
    half_stats = results_df.groupby('half').agg({
        'active_support_ratio': ['mean', 'std'],
        'num_active_supporters': 'mean',
        'event_id': 'count'
    }).round(3)
    half_stats.columns = ['avg_ratio', 'std_ratio', 'avg_active', 'num_events']

    h1_ratios = results_df[results_df['half'] == 'First Half']['active_support_ratio']
    h2_ratios = results_df[results_df['half'] == 'Second Half']['active_support_ratio']
    stat, p_val = stats.mannwhitneyu(h1_ratios, h2_ratios)

    return {
        'half_stats': half_stats,
        'h1_mean': h1_ratios.mean(),
        'h2_mean': h2_ratios.mean(),
        'u_statistic': stat,
        'p_value': p_val,
        'significant': p_val < 0.05
    }


def print_half_comparison(result: dict) -> None:
    """Print formatted half comparison result."""
    print(f"\nFirst Half vs Second Half ASI:")
    print(f"  H1 mean: {result['h1_mean']:.1%}, H2 mean: {result['h2_mean']:.1%}")
    sig_marker = 'Significant' if result['significant'] else '(not significant)'
    print(f"  Mann-Whitney U: {result['u_statistic']:.0f}, p = {result['p_value']:.3f} {sig_marker}")


def calculate_player_fatigue(loader: ASIDataLoader, config: ASIConfig,
                              match_ids: list = None, min_opps: int = 15) -> pd.DataFrame:
    """
    Calculate H1 vs H2 ASI per player to identify fatigue patterns.

    Args:
        loader: ASIDataLoader instance
        config: ASIConfig instance
        match_ids: List of match IDs to process (default: all 10)
        min_opps: Minimum opportunities per half to qualify (default: 15)

    Returns:
        DataFrame with qualified players and fatigue metrics
    """
    if match_ids is None:
        match_ids = [1886347, 1899585, 1925299, 1953632, 1996435,
                     2006229, 2011166, 2013725, 2015213, 2017461]

    player_half_data = []

    for match_id in match_ids:
        try:
            match = loader.load_match(match_id)
            calc = ASICalculator(match, config=config)
            detailed = calc.get_detailed_results()

            # Extract each teammate's contribution per event
            for event in detailed:
                half = 'H1' if event.period == 1 else 'H2'
                for tm in event.teammate_details:
                    player_half_data.append({
                        'match_id': match_id,
                        'player_id': tm['teammate_id'],
                        'player_name': tm['teammate_name'],
                        'half': half,
                        'is_active': 1 if tm['is_active_supporter'] else 0,
                        'is_nearby': 1 if tm['is_nearby'] else 0
                    })
        except Exception as e:
            print(f"Match {match_id}: {e}")

    player_half_df = pd.DataFrame(player_half_data)

    # Aggregate to player + half level (only nearby opportunities count)
    nearby_only = player_half_df[player_half_df['is_nearby'] == 1]
    player_half_asi = nearby_only.groupby(['player_id', 'player_name', 'half']).agg({
        'is_active': ['sum', 'count']
    }).reset_index()
    player_half_asi.columns = ['player_id', 'player_name', 'half', 'active_count', 'opportunities']
    player_half_asi['asi'] = player_half_asi['active_count'] / player_half_asi['opportunities']

    # Pivot to get H1 and H2 side by side
    player_pivot = player_half_asi.pivot(index=['player_id', 'player_name'],
                                          columns='half',
                                          values=['asi', 'opportunities']).reset_index()
    player_pivot.columns = ['player_id', 'player_name', 'h1_asi', 'h2_asi', 'h1_opps', 'h2_opps']

    # Filter: require min_opps in EACH half
    qualified = player_pivot[
        (player_pivot['h1_opps'] >= min_opps) &
        (player_pivot['h2_opps'] >= min_opps)
    ].copy()

    # Calculate fatigue metrics
    qualified['fatigue_drop'] = qualified['h1_asi'] - qualified['h2_asi']  # positive = declined
    qualified['total_opps'] = qualified['h1_opps'] + qualified['h2_opps']
    qualified = qualified.sort_values('fatigue_drop')

    return qualified


def plot_fatigue_comparison(qualified_df: pd.DataFrame, top_n: int = 5,
                             show: bool = True) -> plt.Figure:
    """
    Create side-by-side chart comparing top maintainers vs top faders.

    Args:
        qualified_df: DataFrame from calculate_player_fatigue()
        top_n: Number of players to show on each side
        show: Whether to display the plot

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Top N Maintainers (improved or maintained - lowest fatigue_drop)
    maintainers = qualified_df.head(top_n)
    ax1 = axes[0]
    y_pos = range(len(maintainers))
    ax1.barh(y_pos, maintainers['h1_asi'] * 100, height=0.35, label='H1 ASI', color='#4dabf7', alpha=0.8)
    ax1.barh([y + 0.35 for y in y_pos], maintainers['h2_asi'] * 100, height=0.35, label='H2 ASI', color='#f06595', alpha=0.8)
    ax1.set_yticks([y + 0.175 for y in y_pos])
    ax1.set_yticklabels(maintainers['player_name'].values)
    ax1.set_xlabel('ASI (%)')
    ax1.set_title(f'Top {top_n} Maintainers\n(Improved or held steady)', fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.set_xlim(0, 100)

    # Top N Faders (biggest decline - highest fatigue_drop)
    faders = qualified_df.tail(top_n).iloc[::-1]
    ax2 = axes[1]
    y_pos = range(len(faders))
    ax2.barh(y_pos, faders['h1_asi'] * 100, height=0.35, label='H1 ASI', color='#4dabf7', alpha=0.8)
    ax2.barh([y + 0.35 for y in y_pos], faders['h2_asi'] * 100, height=0.35, label='H2 ASI', color='#f06595', alpha=0.8)
    ax2.set_yticks([y + 0.175 for y in y_pos])
    ax2.set_yticklabels(faders['player_name'].values)
    ax2.set_xlabel('ASI (%)')
    ax2.set_title(f'Top {top_n} Faders\n(Biggest H2 decline)', fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.set_xlim(0, 100)

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def print_fatigue_summary(qualified_df: pd.DataFrame) -> None:
    """Print formatted fatigue analysis summary."""
    n_players = len(qualified_df)
    n_improved = (qualified_df['fatigue_drop'] < 0).sum()
    n_declined = (qualified_df['fatigue_drop'] > 0).sum()
    avg_drop = qualified_df['fatigue_drop'].mean()

    print(f"\nFatigue Analysis Summary ({n_players} players):")
    print(f"  Players who improved H1→H2: {n_improved}")
    print(f"  Players who declined H1→H2: {n_declined}")
    print(f"  Avg fatigue drop: {avg_drop:.1%}")
