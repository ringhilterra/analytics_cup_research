"""
ASI Metrics Calculator

Computes aggregate statistics and derived metrics from ASI analysis:
- Team-level ASI
- Player-level ASI rankings
- Static Rate (team diagnostic)
- By-subtype, by-period, by-zone breakdowns
- Retention correlation (exploratory)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional

from .asi_data_loader import MatchData
from .asi_core import ASICalculator, PressureEventResult


@dataclass
class ASIAggregateStats:
    """Container for aggregate ASI statistics."""
    match_id: int
    total_pressure_events: int
    total_static: int
    static_rate: float  # % of events with 0 active supporters

    # Core metrics
    avg_active_supporters: float
    avg_teammates_nearby: float
    max_supporters_single_event: int

    # Distribution
    events_with_1plus: int
    events_with_2plus: int
    events_with_3plus: int

    # Breakdowns (stored as dicts)
    by_subtype: dict
    by_period: dict
    by_team: dict


class ASIMetrics:
    """
    Calculate aggregate ASI metrics from processed results.

    Provides team diagnostics, player rankings, and situational breakdowns.
    """

    def __init__(self, match_data: MatchData):
        self.match_data = match_data
        self.pitch_zones = self._define_pitch_zones()

    def _define_pitch_zones(self) -> dict:
        """Define pitch zones for spatial analysis."""
        # Zones based on thirds and channels
        L = self.match_data.pitch_length / 2  # Half-length
        W = self.match_data.pitch_width / 2   # Half-width

        return {
            'defensive_third': {'x_min': -L, 'x_max': -L/3},
            'middle_third': {'x_min': -L/3, 'x_max': L/3},
            'attacking_third': {'x_min': L/3, 'x_max': L},
            'left_channel': {'y_min': W/3, 'y_max': W},
            'center_channel': {'y_min': -W/3, 'y_max': W/3},
            'right_channel': {'y_min': -W, 'y_max': -W/3},
        }

    def calculate_aggregate_stats(self, results_df: pd.DataFrame) -> ASIAggregateStats:
        """
        Calculate comprehensive aggregate statistics.

        Args:
            results_df: Pressure events DataFrame from ASICalculator

        Returns:
            ASIAggregateStats object
        """
        total = len(results_df)
        # Static = no active supporters (active_support_ratio == 0)
        static_count = (results_df['active_support_ratio'] == 0).sum()

        # By subtype
        by_subtype = results_df.groupby('pressure_subtype').agg({
            'event_id': 'count',
            'num_active_supporters': 'mean',
            'num_teammates_nearby': 'mean',
            'active_support_ratio': lambda x: (x == 0).sum()  # count static events
        }).rename(columns={
            'event_id': 'count',
            'active_support_ratio': 'static_events',
            'num_active_supporters': 'avg_active',
            'num_teammates_nearby': 'avg_nearby'
        })
        by_subtype['static_rate'] = (by_subtype['static_events'] / by_subtype['count']).round(4)

        # By period
        by_period = results_df.groupby('period').agg({
            'event_id': 'count',
            'num_active_supporters': 'mean',
            'active_support_ratio': [lambda x: (x == 0).sum(), lambda x: (x == 0).mean()]
        })
        by_period.columns = ['count', 'avg_active', 'static_events', 'static_rate']

        # By team
        by_team = {}
        for team_id in results_df['pressed_team_id'].unique():
            if team_id < 0:
                continue
            team_events = results_df[results_df['pressed_team_id'] == team_id]
            team_name = (self.match_data.home_team_name if team_id == self.match_data.home_team_id
                         else self.match_data.away_team_name)
            static_mask = team_events['active_support_ratio'] == 0
            by_team[team_id] = {
                'team_name': team_name,
                'count': len(team_events),
                'static_events': int(static_mask.sum()),
                'static_rate': round(static_mask.mean(), 4),
                'avg_active': round(team_events['num_active_supporters'].mean(), 2),
                'avg_nearby': round(team_events['num_teammates_nearby'].mean(), 2)
            }

        return ASIAggregateStats(
            match_id=self.match_data.match_id,
            total_pressure_events=total,
            total_static=int(static_count),
            static_rate=round(static_count / total, 4) if total > 0 else 0,
            avg_active_supporters=round(results_df['num_active_supporters'].mean(), 2),
            avg_teammates_nearby=round(results_df['num_teammates_nearby'].mean(), 2),
            max_supporters_single_event=int(results_df['num_active_supporters'].max()),
            events_with_1plus=int((results_df['num_active_supporters'] >= 1).sum()),
            events_with_2plus=int((results_df['num_active_supporters'] >= 2).sum()),
            events_with_3plus=int((results_df['num_active_supporters'] >= 3).sum()),
            by_subtype=by_subtype.to_dict('index'),
            by_period=by_period.to_dict('index'),
            by_team=by_team
        )

    def calculate_zone_stats(self, results_df: pd.DataFrame, team_id: int = None) -> pd.DataFrame:
        """
        Calculate ASI metrics by pitch zone for a specific team.

        Zones are labeled relative to the team's attack direction, which is
        read from metadata (home_team_side) for each period.

        Args:
            results_df: Pressure events DataFrame
            team_id: Team to analyze. If None, returns combined stats (legacy behavior,
                     but zone labels will only be correct for home team).

        Returns:
            DataFrame with zone-level stats
        """

        def get_third(x, pressed_team_id, period):
            """Get third relative to team's attack direction."""
            L = self.match_data.pitch_length / 2
            # Get actual attack direction from metadata
            attacks_positive = (
                self.match_data.get_attack_direction(pressed_team_id, period) == 'positive_x'
            )

            if attacks_positive:
                # Team attacks positive x: positive x = attacking third
                if x > L / 3:
                    return 'attacking'
                elif x > -L / 3:
                    return 'middle'
                else:
                    return 'defensive'
            else:
                # Team attacks negative x: negative x = attacking third
                if x < -L / 3:
                    return 'attacking'
                elif x < L / 3:
                    return 'middle'
                else:
                    return 'defensive'

        def get_channel(y, pressed_team_id, period):
            """Get channel relative to team's attack direction."""
            W = self.match_data.pitch_width / 2
            # Get actual attack direction from metadata
            attacks_positive = (
                self.match_data.get_attack_direction(pressed_team_id, period) == 'positive_x'
            )

            if attacks_positive:
                # Team attacks positive x (left = positive y from their perspective)
                if y > W / 3:
                    return 'left'
                elif y > -W / 3:
                    return 'center'
                else:
                    return 'right'
            else:
                # Team attacks negative x (left = negative y from their perspective)
                if y < -W / 3:
                    return 'left'
                elif y < W / 3:
                    return 'center'
                else:
                    return 'right'

        # Filter to specific team if provided
        if team_id is not None:
            df = results_df[results_df['pressed_team_id'] == team_id].copy()
        else:
            df = results_df.copy()

        if len(df) == 0:
            return pd.DataFrame()

        # Apply team-aware zone assignment (using period for correct direction)
        df['third'] = df.apply(
            lambda r: get_third(r['pressed_x'], r['pressed_team_id'], r['period']), axis=1
        )
        df['channel'] = df.apply(
            lambda r: get_channel(r['pressed_y'], r['pressed_team_id'], r['period']), axis=1
        )
        df['zone'] = df['third'] + '_' + df['channel']

        # Aggregate by zone
        zone_stats = df.groupby('zone').agg({
            'event_id': 'count',
            'num_active_supporters': 'mean',
            'num_teammates_nearby': 'mean',
            'active_support_ratio': ['mean', lambda x: (x == 0).sum(), lambda x: (x == 0).mean()]
        })
        zone_stats.columns = ['count', 'avg_active', 'avg_nearby', 'avg_ratio', 'static_events', 'static_rate']
        zone_stats['asi'] = 1 - zone_stats['static_rate']

        return zone_stats.sort_values('count', ascending=False)
