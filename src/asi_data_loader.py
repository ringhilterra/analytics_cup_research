"""
ASI Data Loader

Handles loading and validation of SkillCorner tracking and event data.
Ensures proper frame alignment between pressure events and tracking data.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MatchData:
    """Container for all match data."""
    match_id: int
    tracking_df: pd.DataFrame
    events_df: pd.DataFrame
    metadata: dict

    # Derived data (populated after validation)
    pressure_df: Optional[pd.DataFrame] = None
    player_team_map: dict = field(default_factory=dict)

    # Validation results
    is_validated: bool = False
    validation_errors: list = field(default_factory=list)

    @property
    def home_team_id(self) -> int:
        return self.metadata.get('home_team.id')

    @property
    def away_team_id(self) -> int:
        return self.metadata.get('away_team.id')

    @property
    def home_team_name(self) -> str:
        return self.metadata.get('home_team.name', 'Home')

    @property
    def away_team_name(self) -> str:
        return self.metadata.get('away_team.name', 'Away')

    @property
    def home_team_score(self) -> int:
        return self.metadata.get('home_team_score', 0)

    @property
    def away_team_score(self) -> int:
        return self.metadata.get('away_team_score', 0)

    @property
    def pitch_length(self) -> float:
        return self.metadata.get('pitch_length', 105)

    @property
    def pitch_width(self) -> float:
        return self.metadata.get('pitch_width', 68)

    @property
    def home_team_side(self) -> list:
        """
        Get home team attack direction per half.

        Returns:
            List of ['first_half_direction', 'second_half_direction']
            where direction is 'right_to_left' or 'left_to_right'
        """
        return self.metadata.get('home_team_side', ['right_to_left', 'left_to_right'])

    def get_attack_direction(self, team_id: int, period: int = 1) -> str:
        """
        Get attack direction for a team in a specific period.

        Args:
            team_id: Team ID to query
            period: 1 for first half, 2 for second half

        Returns:
            'positive_x' if team attacks toward positive x,
            'negative_x' if team attacks toward negative x
        """
        # Get home team's direction for this period (0-indexed)
        period_idx = 0 if period == 1 else 1
        home_direction = self.home_team_side[period_idx]

        if team_id == self.home_team_id:
            # Home team direction from metadata
            if home_direction == 'left_to_right':
                return 'positive_x'
            else:  # right_to_left
                return 'negative_x'
        else:
            # Away team attacks opposite direction
            if home_direction == 'left_to_right':
                return 'negative_x'
            else:  # right_to_left
                return 'positive_x'


class ASIDataLoader:
    """
    Loads and validates SkillCorner data for ASI analysis.

    Usage:
        loader = ASIDataLoader(data_dir="./data")
        match_data = loader.load_match(1886347)

        # Or load multiple matches
        all_matches = loader.load_all_matches()
    """

    PRESSURE_SUBTYPES = ('pressure', 'pressing', 'recovery_press', 'counter_press')

    REQUIRED_TRACKING_COLS = ['frame', 'player_id', 'x', 'y', 'speed', 'team_id']
    REQUIRED_EVENT_COLS = ['event_id', 'frame_start', 'frame_end', 'event_type',
                           'event_subtype', 'player_id', 'team_id']

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)

    def get_available_matches(self) -> list[int]:
        """Return list of match IDs available in data directory."""
        matches = []
        if self.data_dir.exists():
            for subdir in self.data_dir.iterdir():
                if subdir.is_dir() and subdir.name.isdigit():
                    # Check if required files exist
                    tracking_file = subdir / f"{subdir.name}_enriched_tracking.csv"
                    events_file = subdir / f"{subdir.name}_dynamic_events.csv"
                    if tracking_file.exists() and events_file.exists():
                        matches.append(int(subdir.name))
        return sorted(matches)

    def load_match(self, match_id: int, validate: bool = True) -> MatchData:
        """
        Load all data for a single match.

        Args:
            match_id: The match ID to load
            validate: Whether to run validation checks

        Returns:
            MatchData object with all loaded data
        """
        match_dir = self.data_dir / str(match_id)

        if not match_dir.exists():
            raise FileNotFoundError(f"Match directory not found: {match_dir}")

        # Load tracking data
        tracking_path = match_dir / f"{match_id}_enriched_tracking.csv"
        tracking_df = pd.read_csv(tracking_path, low_memory=False)

        # Load dynamic events
        events_path = match_dir / f"{match_id}_dynamic_events.csv"
        events_df = pd.read_csv(events_path, low_memory=False)

        # Load metadata
        metadata_path = match_dir / f"{match_id}_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)[0]  # First item in list

        # Create MatchData object
        match_data = MatchData(
            match_id=match_id,
            tracking_df=tracking_df,
            events_df=events_df,
            metadata=metadata
        )

        # Build player-team lookup
        match_data.player_team_map = self._build_player_team_map(events_df)

        # Extract pressure events
        match_data.pressure_df = self._extract_pressure_events(events_df)

        # Validate if requested
        if validate:
            self._validate_match_data(match_data)

        return match_data

    def _build_player_team_map(self, events_df: pd.DataFrame) -> dict:
        """Build mapping from player_id to team_id."""
        return (
            events_df[['player_id', 'team_id']]
            .dropna()
            .drop_duplicates('player_id')
            .set_index('player_id')['team_id']
            .astype(int)
            .to_dict()
        )

    def _extract_pressure_events(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Extract pressure-related on_ball_engagement events."""
        mask = (
            (events_df['event_type'] == 'on_ball_engagement') &
            (events_df['event_subtype'].isin(self.PRESSURE_SUBTYPES))
        )
        return events_df[mask].copy().reset_index(drop=True)

    def _validate_match_data(self, match_data: MatchData) -> None:
        """
        Validate match data integrity and frame alignment.
        Sets is_validated and populates validation_errors.
        """
        errors = []

        # Check required columns in tracking
        missing_tracking = set(self.REQUIRED_TRACKING_COLS) - set(match_data.tracking_df.columns)
        if missing_tracking:
            errors.append(f"Missing tracking columns: {missing_tracking}")

        # Check required columns in events
        missing_events = set(self.REQUIRED_EVENT_COLS) - set(match_data.events_df.columns)
        if missing_events:
            errors.append(f"Missing event columns: {missing_events}")

        # Check for NaN in critical tracking columns
        tracking_nulls = match_data.tracking_df[['frame', 'player_id', 'x', 'y']].isnull().sum()
        if tracking_nulls.any():
            cols_with_nulls = tracking_nulls[tracking_nulls > 0].to_dict()
            errors.append(f"Null values in tracking: {cols_with_nulls}")

        # Check pressure events exist
        if match_data.pressure_df is None or len(match_data.pressure_df) == 0:
            errors.append("No pressure events found")
        else:
            # Validate frame alignment - check that pressure frames exist in tracking
            pressure_frames = set(match_data.pressure_df['frame_start'].dropna().astype(int))
            tracking_frames = set(match_data.tracking_df['frame'].unique())
            missing_frames = pressure_frames - tracking_frames

            if missing_frames:
                # Only error if significant portion missing
                missing_pct = len(missing_frames) / len(pressure_frames) * 100
                if missing_pct > 5:
                    errors.append(f"Frame alignment issue: {len(missing_frames)} pressure frames "
                                  f"({missing_pct:.1f}%) not found in tracking")

        match_data.validation_errors = errors
        match_data.is_validated = len(errors) == 0

    def get_tracking_at_frame(self, match_data: MatchData, frame: int) -> pd.DataFrame:
        """
        Get tracking data for a specific frame.

        Args:
            match_data: The MatchData object
            frame: Frame number to query

        Returns:
            DataFrame with all players at that frame
        """
        return match_data.tracking_df[match_data.tracking_df['frame'] == frame].copy()


def print_match_summary(match_data: MatchData) -> None:
    """Print a summary of loaded match data."""
    print("=" * 60)
    print(f"MATCH {match_data.match_id}")
    print(f"{match_data.home_team_name} {match_data.home_team_score} - {match_data.away_team_score} {match_data.away_team_name}")
    print("-" * 60)
    print(f"Tracking rows:    {len(match_data.tracking_df):,}")
    print(f"Event rows:       {len(match_data.events_df):,}")
    print(f"Pressure events:  {len(match_data.pressure_df):,}")
    print(f"Pitch:            {match_data.pitch_length}m x {match_data.pitch_width}m")
    print(f"Validated:        {'Yes' if match_data.is_validated else 'No'}")
    if match_data.validation_errors:
        print(f"Errors:           {match_data.validation_errors}")
    print("=" * 60)
