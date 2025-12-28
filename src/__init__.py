"""
ASI (Active Support Index) - Off-Ball Movement Analysis

Quantifies player support during high-pressure moments by measuring
teammate movement velocity when a ball carrier is under pressure.
"""

__version__ = "0.1.0"

# Lazy imports - modules imported when accessed
def __getattr__(name):
    if name == "ASIDataLoader":
        from .asi_data_loader import ASIDataLoader
        return ASIDataLoader
    elif name == "MatchData":
        from .asi_data_loader import MatchData
        return MatchData
    elif name == "ASICalculator":
        from .asi_core import ASICalculator
        return ASICalculator
    elif name == "ASIConfig":
        from .asi_core import ASIConfig
        return ASIConfig
    elif name == "ASIMetrics":
        from .asi_metrics import ASIMetrics
        return ASIMetrics
    elif name == "ASIVisualizer":
        from .asi_visualizations import ASIVisualizer
        return ASIVisualizer
    elif name == "fetch_match_data":
        from .asi_data_fetcher import fetch_match_data
        return fetch_match_data
    elif name == "get_available_match_ids":
        from .asi_data_fetcher import get_available_match_ids
        return get_available_match_ids
    elif name == "process_all_matches":
        from .asi_multi_match import process_all_matches
        return process_all_matches
    elif name == "get_top_players_all_matches":
        from .asi_multi_match import get_top_players_all_matches
        return get_top_players_all_matches
    elif name == "get_team_stats_all_matches":
        from .asi_multi_match import get_team_stats_all_matches
        return get_team_stats_all_matches
    elif name == "plot_multi_match_team_comparison":
        from .asi_visualizations import plot_multi_match_team_comparison
        return plot_multi_match_team_comparison
    # Validation functions
    elif name == "categorize_position":
        from .asi_validation import categorize_position
        return categorize_position
    elif name == "calculate_position_stats":
        from .asi_validation import calculate_position_stats
        return calculate_position_stats
    elif name == "plot_position_validation":
        from .asi_validation import plot_position_validation
        return plot_position_validation
    elif name == "test_position_significance":
        from .asi_validation import test_position_significance
        return test_position_significance
    elif name == "print_significance_result":
        from .asi_validation import print_significance_result
        return print_significance_result
    # Time analysis functions
    elif name == "calculate_time_based_asi":
        from .asi_time_analysis import calculate_time_based_asi
        return calculate_time_based_asi
    elif name == "get_time_bin_stats":
        from .asi_time_analysis import get_time_bin_stats
        return get_time_bin_stats
    elif name == "plot_time_trend":
        from .asi_time_analysis import plot_time_trend
        return plot_time_trend
    elif name == "compare_halves":
        from .asi_time_analysis import compare_halves
        return compare_halves
    elif name == "print_half_comparison":
        from .asi_time_analysis import print_half_comparison
        return print_half_comparison
    elif name == "calculate_player_fatigue":
        from .asi_time_analysis import calculate_player_fatigue
        return calculate_player_fatigue
    elif name == "plot_fatigue_comparison":
        from .asi_time_analysis import plot_fatigue_comparison
        return plot_fatigue_comparison
    elif name == "print_fatigue_summary":
        from .asi_time_analysis import print_fatigue_summary
        return print_fatigue_summary
    # Physical validation functions
    elif name == "load_physical_aggregates":
        from .asi_physical_validation import load_physical_aggregates
        return load_physical_aggregates
    elif name == "merge_asi_with_physical":
        from .asi_physical_validation import merge_asi_with_physical
        return merge_asi_with_physical
    elif name == "calculate_physical_correlation":
        from .asi_physical_validation import calculate_physical_correlation
        return calculate_physical_correlation
    elif name == "plot_asi_physical_correlation":
        from .asi_physical_validation import plot_asi_physical_correlation
        return plot_asi_physical_correlation
    elif name == "print_physical_validation_summary":
        from .asi_physical_validation import print_physical_validation_summary
        return print_physical_validation_summary
    raise AttributeError(f"module 'src' has no attribute '{name}'")
