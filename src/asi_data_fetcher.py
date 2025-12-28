"""
ASI Data Fetcher - Minimal data fetcher for ASI analysis.

Fetches tracking and event data from SkillCorner GitHub with only
the columns needed for ASI calculations. Skips pass processing.

Usage:
    from src import fetch_match_data
    fetch_match_data(1886347)  # Downloads and processes match data to ./data/1886347/
"""

import pandas as pd
import numpy as np
import requests
from pathlib import Path
from scipy.signal import savgol_filter


# SkillCorner GitHub URLs
SKILLCORNER_BASE = "https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches"
SKILLCORNER_MEDIA = "https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches"


def _calculate_speed(series, fps=10.0, window=7, poly=2):
    """Calculate speed using Savitzky-Golay filter derivative."""
    if len(series) < window:
        return np.zeros(len(series))
    return savgol_filter(series, window, poly, deriv=1, delta=1.0/fps)


def _fetch_metadata(match_id: int) -> tuple[dict, pd.DataFrame]:
    """
    Fetch and process match metadata.

    Returns:
        Tuple of (raw metadata dict, players DataFrame with minimal columns)
    """
    url = f"{SKILLCORNER_BASE}/{match_id}/{match_id}_match.json"
    print(f"  Fetching metadata from: {url}")

    resp = requests.get(url)
    resp.raise_for_status()
    meta_data = resp.json()

    # Flatten metadata
    raw_match_df = pd.json_normalize(meta_data, max_level=2)

    # Extract player data
    players_df = pd.json_normalize(
        raw_match_df.to_dict("records"),
        record_path="players",
        meta=["home_team.name", "home_team.id", "away_team.name", "away_team.id"]
    )

    # Filter for players who actually played
    players_df = players_df[
        ~((players_df.start_time.isna()) & (players_df.end_time.isna()))
    ].copy()

    # Keep only columns needed for ASI (including position for context)
    players_cols = ['id', 'short_name', 'number', 'team_id', 'player_role.acronym', 'player_role.name']
    players_df = players_df[[c for c in players_cols if c in players_df.columns]]

    return meta_data, players_df


def _fetch_tracking(match_id: int) -> pd.DataFrame:
    """
    Fetch and process tracking data with speed calculation.

    Returns:
        DataFrame with columns: frame, player_id, x, y, speed, period, ball_x, ball_y
    """
    url = f"{SKILLCORNER_MEDIA}/{match_id}/{match_id}_tracking_extrapolated.jsonl"
    print(f"  Fetching tracking from: {url}")

    raw_tracking = pd.read_json(url, lines=True)

    # Flatten player data
    tracking_df = pd.json_normalize(
        raw_tracking.to_dict("records"),
        record_path="player_data",
        meta=["frame", "timestamp", "period", "ball_data"]
    )

    # Extract ball position
    ball_norm = pd.json_normalize(tracking_df["ball_data"])
    tracking_df["ball_x"] = ball_norm["x"] if "x" in ball_norm.columns else np.nan
    tracking_df["ball_y"] = ball_norm["y"] if "y" in ball_norm.columns else np.nan
    tracking_df = tracking_df.drop(columns=["ball_data"])

    # Calculate match time from timestamp
    tracking_df["timestamp"] = pd.to_datetime(tracking_df["timestamp"])
    tracking_df["period_start_time"] = tracking_df.groupby("period")["timestamp"].transform("min")
    tracking_df["elapsed_s"] = (tracking_df["timestamp"] - tracking_df["period_start_time"]).dt.total_seconds()
    tracking_df["elapsed_min"] = tracking_df["elapsed_s"] / 60
    tracking_df["match_time_min"] = np.where(
        tracking_df["period"] == 1,
        tracking_df["elapsed_min"],
        tracking_df["elapsed_min"] + 45
    )
    # Drop intermediate columns
    tracking_df = tracking_df.drop(columns=["timestamp", "period_start_time", "elapsed_s", "elapsed_min"])

    # Sort for velocity calculation
    tracking_df = tracking_df.sort_values(['player_id', 'period', 'frame'])

    # Calculate velocity components and speed (all in m/s)
    # Savitzky-Golay filter computes derivative of position over time (fps=10 → delta=0.1s)
    print("  Calculating player velocities...")
    tracking_df['vx'] = tracking_df.groupby(['player_id', 'period'])['x'].transform(_calculate_speed)
    tracking_df['vy'] = tracking_df.groupby(['player_id', 'period'])['y'].transform(_calculate_speed)
    tracking_df['speed'] = np.sqrt(tracking_df['vx']**2 + tracking_df['vy']**2)

    # Keep only columns needed by ASI
    tracking_cols = ['frame', 'player_id', 'x', 'y', 'vx', 'vy', 'speed', 'period', 'match_time_min', 'ball_x', 'ball_y']
    tracking_df = tracking_df[[c for c in tracking_cols if c in tracking_df.columns]]

    return tracking_df.sort_values(['period', 'frame', 'player_id'])


def _fetch_events(match_id: int) -> pd.DataFrame:
    """
    Fetch dynamic events (no coordinate transformations needed for ASI).

    ASI only uses event metadata to identify pressure events,
    then looks up positions from tracking data.

    Returns:
        DataFrame with event columns needed for ASI
    """
    url = f"{SKILLCORNER_BASE}/{match_id}/{match_id}_dynamic_events.csv"
    print(f"  Fetching events from: {url}")

    events_df = pd.read_csv(url)
    events_df.columns = [c.lower() for c in events_df.columns]

    # Keep only columns needed by ASI
    # Core columns for pressure event identification
    asi_cols = [
        'event_id', 'frame_start', 'frame_end', 'event_type', 'event_subtype',
        'player_id', 'team_id', 'player_name', 'period',
        'player_in_possession_id', 'player_in_possession_name',
        # For sequence detection
        'phase_index', 'team_possession_loss_in_phase',
        'associated_player_possession_end_type'
    ]

    events_df = events_df[[c for c in asi_cols if c in events_df.columns]]

    return events_df


def _enrich_tracking(tracking_df: pd.DataFrame, players_df: pd.DataFrame) -> pd.DataFrame:
    """Merge tracking data with player metadata."""
    print("  Enriching tracking with player info...")

    enriched = tracking_df.merge(
        players_df,
        left_on="player_id",
        right_on="id",
        how="left"
    )

    # Drop duplicate id column if present
    if 'id' in enriched.columns:
        enriched = enriched.drop(columns=['id'])

    return enriched


def fetch_match_data(
    match_id: int,
    output_dir: str = "./data",
    verbose: bool = True,
    skip_if_exists: bool = True
) -> Path:
    """
    Fetch and process SkillCorner data for ASI analysis.

    Downloads tracking, events, and metadata from SkillCorner GitHub,
    processes them with minimal columns needed for ASI, and saves to disk.

    Args:
        match_id: SkillCorner match ID
        output_dir: Base output directory (default: ./data)
        verbose: Print progress messages
        skip_if_exists: Skip fetching if data already exists (default True)

    Returns:
        Path to match data directory

    Example:
        >>> from src import fetch_match_data, ASIDataLoader
        >>> fetch_match_data(1886347)
        >>> loader = ASIDataLoader()
        >>> match_data = loader.load_match(1886347)
    """
    # Check if data already exists
    match_dir = Path(output_dir) / str(match_id)
    tracking_file = match_dir / f"{match_id}_enriched_tracking.csv"

    if skip_if_exists and tracking_file.exists():
        if verbose:
            print(f"Match {match_id} data already exists, skipping fetch.")
        return match_dir

    if verbose:
        print(f"\n{'='*60}")
        print(f"ASI Data Fetcher - Match {match_id}")
        print(f"{'='*60}")

    # Create output directory
    match_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Fetch metadata
        meta_data, players_df = _fetch_metadata(match_id)

        # 2. Fetch and process tracking
        tracking_df = _fetch_tracking(match_id)

        # 3. Enrich tracking with player info
        enriched_tracking = _enrich_tracking(tracking_df, players_df)

        # 4. Fetch events
        events_df = _fetch_events(match_id)

        # 5. Save files
        if verbose:
            print(f"\n  Saving files to {match_dir}/")

        # Enriched tracking
        tracking_path = match_dir / f"{match_id}_enriched_tracking.csv"
        enriched_tracking.to_csv(tracking_path, index=False)
        if verbose:
            print(f"    {tracking_path.name}: {len(enriched_tracking):,} rows")

        # Dynamic events
        events_path = match_dir / f"{match_id}_dynamic_events.csv"
        events_df.to_csv(events_path, index=False)
        if verbose:
            print(f"    {events_path.name}: {len(events_df):,} rows")

        # Metadata (as JSON array for compatibility with ASIDataLoader)
        meta_path = match_dir / f"{match_id}_metadata.json"
        meta_df = pd.json_normalize(meta_data, max_level=2)
        meta_df.to_json(meta_path, orient='records', indent=4)
        if verbose:
            print(f"    {meta_path.name}")

        if verbose:
            print(f"\n{'='*60}")
            print(f"Done! Data saved to {match_dir}/")
            print(f"{'='*60}\n")

        return match_dir

    except requests.exceptions.HTTPError as e:
        print(f"\nError fetching data: {e}")
        print("Check that the match_id is valid in the SkillCorner open data repository.")
        raise
    except Exception as e:
        print(f"\nError processing data: {e}")
        raise


def get_available_match_ids() -> list[int]:
    """
    Get list of match IDs available in SkillCorner open data.

    Returns:
        List of 10 match IDs from the A-League dataset.
    """
    # Known match IDs from SkillCorner open data (A-League dataset)
    # https://github.com/SkillCorner/opendata/tree/master/data/matches
    known_ids = [
        1886347, 1899585, 1925299, 1953632, 1996435,
        2006229, 2011166, 2013725, 2015213, 2017461
    ]
    return known_ids


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        match_id = int(sys.argv[1])
    else:
        # Default to first available match
        match_id = 1886347

    fetch_match_data(match_id)
