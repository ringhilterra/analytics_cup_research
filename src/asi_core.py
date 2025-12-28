"""
ASI Core Calculator

Implements the Active Support Index calculation:
ASI = (Pressure Events with Active Movement > threshold) / (Total Nearby Pressure Events)

Active movement = velocity > velocity_threshold_ms (default: 2 m/s)
Nearby = distance < proximity_threshold_m (default: 35m)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional

from .asi_data_loader import MatchData


@dataclass
class ASIConfig:
    """Configuration for ASI calculation thresholds."""
    proximity_threshold_m: float = 35.0    # Max distance to be "nearby" (from spec)
    velocity_threshold_ms: float = 2.0     # Min speed for "active" support (from spec)
    pressure_subtypes: tuple = ('pressure', 'pressing', 'recovery_press', 'counter_press')


@dataclass
class PressureEventResult:
    """Result of analyzing a single pressure event."""
    event_id: str
    event_idx: int
    frame_start: int
    frame_end: int
    period: int
    pressure_subtype: str

    # Pressed player info
    pressed_player_id: int
    pressed_player_name: str
    pressed_team_id: int
    pressed_x: float
    pressed_y: float

    # Pressing player/team info
    pressing_player_id: int
    pressing_player_name: str
    pressing_team_id: int

    # Support metrics
    num_teammates_nearby: int
    num_active_supporters: int
    active_support_ratio: float  # active_supporters / nearby_teammates (0 if no nearby)

    # Detailed teammate data
    teammate_details: list


class ASICalculator:
    """
    Calculate Active Support Index from tracking and event data.

    The ASI measures how often teammates provide active support (velocity > threshold)
    when a ball carrier is under pressure.

    Usage:
        loader = ASIDataLoader()
        match_data = loader.load_match(1886347)

        calculator = ASICalculator(match_data)
        results = calculator.process_all_pressure_events()
        player_scores = calculator.calculate_player_asi_scores(results)
    """

    def __init__(self, match_data: MatchData, config: Optional[ASIConfig] = None):
        self.match_data = match_data
        self.config = config or ASIConfig()

        # Pre-build frame index for faster lookups
        self._tracking_by_frame = {}
        self._build_frame_index()

    def _build_frame_index(self) -> None:
        """Pre-index tracking data by frame for pressure event frames."""
        pressure_frames = set()
        for _, p in self.match_data.pressure_df.iterrows():
            pressure_frames.add(int(p['frame_start']))

        # Filter tracking to only frames we need
        relevant_tracking = self.match_data.tracking_df[
            self.match_data.tracking_df['frame'].isin(pressure_frames)
        ]

        self._tracking_by_frame = {
            frame: group for frame, group in relevant_tracking.groupby('frame')
        }

    def get_tracking_at_frame(self, frame: int) -> pd.DataFrame:
        """Get tracking data for a specific frame (cached)."""
        if frame in self._tracking_by_frame:
            return self._tracking_by_frame[frame]
        # Fallback to direct query
        return self.match_data.tracking_df[self.match_data.tracking_df['frame'] == frame]

    def get_player_team(self, player_id: float) -> Optional[int]:
        """Get team_id for a player."""
        if pd.isna(player_id):
            return None
        return self.match_data.player_team_map.get(int(player_id))

    def analyze_pressure_event(self, event_idx: int) -> PressureEventResult:
        """
        Analyze a single pressure event.

        Finds all teammates of the pressed player, calculates their distance
        and velocity, and determines who is providing active support.

        Args:
            event_idx: Index into pressure_df

        Returns:
            PressureEventResult with all metrics
        """
        event = self.match_data.pressure_df.iloc[event_idx]
        frame = int(event['frame_start'])

        # Get tracking data at this frame
        frame_data = self.get_tracking_at_frame(frame)

        # Get pressed player info
        pressed_player_id = event['player_in_possession_id']
        pressed_team_id = self.get_player_team(pressed_player_id)

        # Find pressed player position in tracking
        pressed_pos = frame_data[frame_data['player_id'] == pressed_player_id]

        teammate_details = []
        num_nearby = 0
        num_active = 0

        if len(pressed_pos) > 0 and pressed_team_id is not None:
            px, py = pressed_pos['x'].iloc[0], pressed_pos['y'].iloc[0]

            # Find all teammates (same team, excluding pressed player)
            teammates = frame_data[
                (frame_data['team_id'] == pressed_team_id) &
                (frame_data['player_id'] != pressed_player_id)
            ]

            for _, tm in teammates.iterrows():
                # Calculate distance to pressed player
                distance = np.sqrt((tm['x'] - px) ** 2 + (tm['y'] - py) ** 2)

                # Get speed (handle missing values)
                speed = tm['speed'] if pd.notna(tm.get('speed')) else 0.0

                # Apply thresholds
                is_nearby = distance < self.config.proximity_threshold_m
                is_active = speed > self.config.velocity_threshold_ms
                is_active_supporter = is_nearby and is_active

                if is_nearby:
                    num_nearby += 1
                if is_active_supporter:
                    num_active += 1

                teammate_details.append({
                    'teammate_id': int(tm['player_id']),
                    'teammate_name': tm.get('short_name', f"Player {int(tm['player_id'])}"),
                    'x': round(tm['x'], 2),
                    'y': round(tm['y'], 2),
                    'distance_m': round(distance, 2),
                    'speed_ms': round(speed, 2),
                    'is_nearby': is_nearby,
                    'is_active': is_active,
                    'is_active_supporter': is_active_supporter
                })
        else:
            px, py = np.nan, np.nan

        # Calculate active support ratio
        active_support_ratio = num_active / num_nearby if num_nearby > 0 else 0.0

        return PressureEventResult(
            event_id=str(event['event_id']),
            event_idx=event_idx,
            frame_start=frame,
            frame_end=int(event['frame_end']),
            period=int(event['period']),
            pressure_subtype=event['event_subtype'],

            pressed_player_id=int(pressed_player_id) if pd.notna(pressed_player_id) else -1,
            pressed_player_name=str(event.get('player_in_possession_name', 'Unknown')),
            pressed_team_id=int(pressed_team_id) if pressed_team_id else -1,
            pressed_x=px,
            pressed_y=py,

            pressing_player_id=int(event['player_id']) if pd.notna(event['player_id']) else -1,
            pressing_player_name=str(event.get('player_name', 'Unknown')),
            pressing_team_id=int(event['team_id']) if pd.notna(event['team_id']) else -1,

            num_teammates_nearby=num_nearby,
            num_active_supporters=num_active,
            active_support_ratio=round(active_support_ratio, 4),

            teammate_details=teammate_details
        )

    def process_all_pressure_events(self, verbose: bool = False) -> pd.DataFrame:
        """
        Process all pressure events and return summary DataFrame.

        Args:
            verbose: Print progress updates

        Returns:
            DataFrame with one row per pressure event
        """
        results = []
        total = len(self.match_data.pressure_df)

        for idx in range(total):
            if verbose and idx % 100 == 0:
                print(f"Processing pressure event {idx}/{total}...")

            result = self.analyze_pressure_event(idx)
            results.append({
                'event_id': result.event_id,
                'event_idx': result.event_idx,
                'frame_start': result.frame_start,
                'frame_end': result.frame_end,
                'period': result.period,
                'pressure_subtype': result.pressure_subtype,
                'pressed_player_id': result.pressed_player_id,
                'pressed_player_name': result.pressed_player_name,
                'pressed_team_id': result.pressed_team_id,
                'pressed_x': result.pressed_x,
                'pressed_y': result.pressed_y,
                'pressing_player_id': result.pressing_player_id,
                'pressing_player_name': result.pressing_player_name,
                'pressing_team_id': result.pressing_team_id,
                'num_teammates_nearby': result.num_teammates_nearby,
                'num_active_supporters': result.num_active_supporters,
                'active_support_ratio': result.active_support_ratio
            })

        return pd.DataFrame(results)

    def get_detailed_results(self) -> list[PressureEventResult]:
        """Get full results including teammate details for each pressure event."""
        return [self.analyze_pressure_event(idx) for idx in range(len(self.match_data.pressure_df))]

    def calculate_player_asi_scores(self, detailed_results: list[PressureEventResult] = None) -> pd.DataFrame:
        """
        Calculate ASI score for each player.

        ASI = (Times provided active support) / (Opportunities = times nearby when teammate pressed)

        Args:
            detailed_results: List of PressureEventResult (computed if not provided)

        Returns:
            DataFrame with player ASI scores including team_name, player_number, player_role_acronym
        """
        if detailed_results is None:
            detailed_results = self.get_detailed_results()

        # Build player info lookup from tracking data
        player_info = self.match_data.tracking_df.drop_duplicates('player_id')[
            ['player_id', 'number', 'team_id', 'player_role.acronym']
        ].set_index('player_id').to_dict('index')

        # Create team name lookup
        team_names = {
            self.match_data.home_team_id: self.match_data.home_team_name,
            self.match_data.away_team_id: self.match_data.away_team_name
        }

        # Count active support provided and opportunities for each player
        active_counts = {}
        opportunity_counts = {}

        for result in detailed_results:
            for tm in result.teammate_details:
                tm_id = tm['teammate_id']
                tm_name = tm['teammate_name']

                # Only count if nearby (this is an "opportunity" to support)
                if tm['is_nearby']:
                    if tm_id not in opportunity_counts:
                        opportunity_counts[tm_id] = {'name': tm_name, 'count': 0}
                    opportunity_counts[tm_id]['count'] += 1

                    # Count if they provided active support
                    if tm['is_active_supporter']:
                        if tm_id not in active_counts:
                            active_counts[tm_id] = 0
                        active_counts[tm_id] += 1

        # Build player scores DataFrame
        player_scores = []
        for player_id, opp_data in opportunity_counts.items():
            active = active_counts.get(player_id, 0)
            opportunities = opp_data['count']
            asi_score = active / opportunities if opportunities > 0 else 0

            # Get player metadata
            player_info_data = player_info.get(player_id, {})
            team_id = player_info_data.get('team_id')
            number = player_info_data.get('number')

            # Build match name string
            match_name = (f"{self.match_data.home_team_name} vs {self.match_data.away_team_name} "
                          f"({self.match_data.home_team_score}-{self.match_data.away_team_score})")

            player_scores.append({
                'player_id': player_id,
                'player_name': opp_data['name'],
                'team_name': team_names.get(team_id, 'Unknown'),
                'player_number': int(number) if pd.notna(number) else 0,
                'player_role_acronym': player_info_data.get('player_role.acronym', 'N/A'),
                'active_support_count': active,
                'opportunities': opportunities,
                'asi_score': round(asi_score, 4),
                'match_name': match_name,
                'matches_count': 1
            })

        df = pd.DataFrame(player_scores)
        return df.sort_values('asi_score', ascending=False).reset_index(drop=True)

    def calculate_team_asi_scores(self, results_df: pd.DataFrame = None) -> dict:
        """
        Calculate team-level ASI metrics.

        Args:
            results_df: Pressure events DataFrame (computed if not provided)

        Returns:
            Dict with team ASI stats
        """
        if results_df is None:
            results_df = self.process_all_pressure_events()

        team_stats = {}

        for team_id in [self.match_data.home_team_id, self.match_data.away_team_id]:
            # Filter for events where this team was pressed
            team_events = results_df[results_df['pressed_team_id'] == team_id]

            if len(team_events) == 0:
                continue

            total_events = len(team_events)
            static_events = (team_events['active_support_ratio'] == 0).sum()
            avg_supporters = team_events['num_active_supporters'].mean()
            avg_nearby = team_events['num_teammates_nearby'].mean()

            # Team ASI = 1 - static_rate (higher = better support)
            team_asi = 1 - (static_events / total_events)

            team_name = (self.match_data.home_team_name if team_id == self.match_data.home_team_id
                         else self.match_data.away_team_name)

            team_stats[team_id] = {
                'team_name': team_name,
                'total_pressure_events': total_events,
                'static_events': int(static_events),
                'static_rate': round(static_events / total_events, 4),
                'team_asi': round(team_asi, 4),
                'avg_active_supporters': round(avg_supporters, 2),
                'avg_teammates_nearby': round(avg_nearby, 2)
            }

        return team_stats
