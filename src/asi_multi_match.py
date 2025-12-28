"""
ASI Multi-Match Analysis

Functions for running ASI calculations across multiple matches and aggregating results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from .asi_data_loader import ASIDataLoader
from .asi_core import ASICalculator, ASIConfig
from .asi_data_fetcher import get_available_match_ids, fetch_match_data


def process_all_matches(
    data_dir: str = "./data",
    config: Optional[ASIConfig] = None,
    verbose: bool = True,
    auto_fetch: bool = True
) -> dict:
    """
    Run ASI calculation on all available matches.

    Args:
        data_dir: Directory containing match data
        config: Optional ASIConfig (uses defaults if not provided)
        verbose: Print progress updates
        auto_fetch: Automatically download missing match data (default True)

    Returns:
        Dict containing:
        - all_player_scores: Combined player scores with match_id column
        - all_team_stats: Combined team stats with match_id column
        - all_results: All pressure events with match_id
        - match_summaries: List of per-match summary dicts
    """
    loader = ASIDataLoader(data_dir)
    match_ids = get_available_match_ids()
    config = config or ASIConfig()

    # Auto-fetch missing data if enabled
    if auto_fetch:
        for match_id in match_ids:
            match_dir = Path(data_dir) / str(match_id)
            if not match_dir.exists():
                if verbose:
                    print(f"Fetching data for match {match_id}...")
                fetch_match_data(match_id, output_dir=data_dir, verbose=False)

    all_player_scores = []
    all_team_stats = []
    all_results = []
    match_summaries = []

    for i, match_id in enumerate(match_ids):
        if verbose:
            print(f"Processing match {i+1}/{len(match_ids)}: {match_id}...", end=" ")

        try:
            match_data = loader.load_match(match_id)
            calculator = ASICalculator(match_data, config)

            # Get results
            results_df = calculator.process_all_pressure_events()
            results_df['match_id'] = match_id
            all_results.append(results_df)

            # Get player scores
            detailed_results = calculator.get_detailed_results()
            player_scores = calculator.calculate_player_asi_scores(detailed_results)
            player_scores['match_id'] = match_id
            all_player_scores.append(player_scores)

            # Get team stats
            team_stats = calculator.calculate_team_asi_scores(results_df)
            for team_id, stats in team_stats.items():
                stats['team_id'] = team_id
                stats['match_id'] = match_id
                all_team_stats.append(stats)

            # Summary
            match_summaries.append({
                'match_id': match_id,
                'home_team': match_data.home_team_name,
                'away_team': match_data.away_team_name,
                'score': f"{match_data.home_team_score}-{match_data.away_team_score}",
                'pressure_events': len(results_df),
                'players_analyzed': len(player_scores)
            })

            if verbose:
                print(f"{match_data.home_team_name} vs {match_data.away_team_name}")

        except Exception as e:
            if verbose:
                print(f"ERROR: {e}")
            continue

    if verbose:
        print(f"\nProcessed {len(match_summaries)} matches successfully.")

    return {
        'all_player_scores': pd.concat(all_player_scores, ignore_index=True) if all_player_scores else pd.DataFrame(),
        'all_team_stats': pd.DataFrame(all_team_stats),
        'all_results': pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame(),
        'match_summaries': match_summaries
    }


def get_top_players_all_matches(
    all_player_scores: pd.DataFrame,
    min_opportunities: int = 50,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Aggregate player ASI across all matches and rank.

    Args:
        all_player_scores: Combined player scores from process_all_matches()
        min_opportunities: Minimum total opportunities to include
        top_n: Number of top players to return

    Returns:
        DataFrame with aggregated player stats sorted by ASI
    """
    if all_player_scores.empty:
        return pd.DataFrame()

    # Group by player_id and aggregate
    agg = all_player_scores.groupby('player_id').agg({
        'player_name': 'first',
        'team_name': 'first',
        'player_number': 'first',
        'player_role_acronym': 'first',
        'active_support_count': 'sum',
        'opportunities': 'sum',
        'match_id': 'nunique'
    }).reset_index()

    agg.columns = ['player_id', 'player_name', 'team_name', 'player_number',
                   'player_role_acronym', 'active_support_count', 'opportunities',
                   'matches_count']

    # Calculate overall ASI
    agg['asi_score'] = agg['active_support_count'] / agg['opportunities']
    agg['asi_score'] = agg['asi_score'].round(4)

    # Set match_name to empty for aggregated data (multi-match)
    agg['match_name'] = ''

    # Filter and sort
    agg = agg[agg['opportunities'] >= min_opportunities]
    agg = agg.sort_values('asi_score', ascending=False).head(top_n)

    return agg.reset_index(drop=True)


def get_team_stats_all_matches(all_team_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate team stats across all matches.

    Args:
        all_team_stats: Combined team stats from process_all_matches()

    Returns:
        DataFrame with aggregated team stats sorted by avg ASI
    """
    if all_team_stats.empty:
        return pd.DataFrame()

    # Group by team_name and aggregate
    agg = all_team_stats.groupby('team_name').agg({
        'total_pressure_events': 'sum',
        'static_events': 'sum',
        'avg_active_supporters': 'mean',
        'avg_teammates_nearby': 'mean',
        'match_id': 'nunique'
    }).reset_index()

    agg.columns = ['team_name', 'total_pressure_events', 'total_static_events',
                   'avg_active_supporters', 'avg_teammates_nearby', 'matches_played']

    # Calculate overall stats
    agg['overall_static_rate'] = agg['total_static_events'] / agg['total_pressure_events']
    agg['overall_team_asi'] = 1 - agg['overall_static_rate']

    # Round for display
    agg['overall_static_rate'] = agg['overall_static_rate'].round(4)
    agg['overall_team_asi'] = agg['overall_team_asi'].round(4)
    agg['avg_active_supporters'] = agg['avg_active_supporters'].round(2)
    agg['avg_teammates_nearby'] = agg['avg_teammates_nearby'].round(2)

    # Sort by ASI
    agg = agg.sort_values('overall_team_asi', ascending=False)

    return agg.reset_index(drop=True)
