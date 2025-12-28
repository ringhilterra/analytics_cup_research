from typing import List
from .asi_core import PressureEventResult
from .asi_data_loader import ASIDataLoader, MatchData
from .asi_visualizations import ASIVisualizer


def get_event_by_id(detailed_results: List[PressureEventResult], event_id: str) -> PressureEventResult:
    """Look up a pressure event by its event_id."""
    for event in detailed_results:
        if event.event_id == event_id:
            return event
    raise ValueError(f"Event ID '{event_id}' not found.")


def get_detail_results_summary(event: PressureEventResult):
    print("\nDetailed Analysis - Single Pressure Event")
    print("=" * 60)
    print(f"Event ID: {event.event_id}")
    print(f"Frame: {event.frame_start}")
    print(f"Type: {event.pressure_subtype}")
    print(f"\nPressed Player: {event.pressed_player_name}")
    print(f"Position: ({event.pressed_x:.1f}, {event.pressed_y:.1f})")
    print(f"\nSupport Metrics:")
    print(f"  Teammates Nearby (<35m): {event.num_teammates_nearby}")
    print(f"  Active Supporters (>2m/s): {event.num_active_supporters}")
    print(f"  Active Support Ratio: {event.active_support_ratio}")

    print(f"\nTeammate Details:")
    for tm in event.teammate_details[:5]:  # First 5
        status = 'ACTIVE' if tm['is_active_supporter'] else ('STATIC' if tm['is_nearby'] else 'FAR')
        print(f"  {tm['teammate_name']:15} | Dist: {tm['distance_m']:5.1f}m | Speed: {tm['speed_ms']:.1f} m/s | {status}")


def team_stat_asi_summary(team_stats):
    print("Team ASI Comparison")
    print("=" * 60)

    for stats in team_stats.values():
        print(f"\n{stats['team_name']}:")
        print(f"  Total pressure events (when pressed): {stats['total_pressure_events']}")
        print(f"  Team ASI (1 - static rate):           {stats['team_asi']:.1%}")
        print(f"  Static Rate (0 active supporters):   {stats['static_rate']:.1%}")
        print(f"  Avg Active Supporters:                {stats['avg_active_supporters']:.2f}")
        print(f"  Avg Teammates Nearby:                 {stats['avg_teammates_nearby']:.2f}")


def plot_pressure_from_event_id(event_id: str, detailed_results: List[PressureEventResult],
                                  match_data: MatchData, loader: ASIDataLoader) -> None:
    """
    Plot a pressure moment visualization for a given event ID.

    Args:
        event_id: The event ID to visualize
        detailed_results: List of PressureEventResult from calculator.get_detailed_results()
        match_data: MatchData instance
        loader: ASIDataLoader instance
    """
    selected_event = get_event_by_id(detailed_results, event_id)
    frame_data = loader.get_tracking_at_frame(match_data, selected_event.frame_start)
    visualizer = ASIVisualizer(match_data)
    visualizer.plot_pressure_moment(selected_event, frame_data, show=True)