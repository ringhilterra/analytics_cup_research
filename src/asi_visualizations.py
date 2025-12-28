"""
ASI Visualizations

Creates visualizations for ASI analysis using matplotlib, seaborn, and mplsoccer.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mplsoccer import Pitch
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from .asi_data_loader import MatchData
from .asi_core import ASICalculator, ASIConfig, PressureEventResult


class ASIVisualizer:
    """
    Create ASI visualizations.

    Provides methods for:
    - Pressure moment snapshots (pitch view)
    - Player ASI leaderboard (bar chart)
    - Pitch zone heatmaps
    - Sequence timelines
    """

    # Color scheme
    COLORS = {
        'active_supporter': '#00ff00',      # Bright green
        'static_teammate': '#ff4444',       # Red
        'pressed_player': '#ffff00',        # Yellow
        'opponent': '#888888',              # Gray
        'ball': 'white',
        'pitch_background': '#2e8b2e',
        'pitch_lines': 'white',
        'home_team': '#1e90ff',             # Blue
        'away_team': '#dc143c',             # Red
    }

    def __init__(self, match_data: MatchData, output_dir: str = "./output/visualizations",
                 config: Optional[ASIConfig] = None):
        self.match_data = match_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or ASIConfig()

        # Set style
        plt.style.use('dark_background')
        sns.set_palette("husl")

    def plot_pressure_moment(self, pressure_result: PressureEventResult,
                              frame_data: pd.DataFrame,
                              save_path: Optional[str] = None,
                              show: bool = True) -> plt.Figure:
        """
        Visualize a single pressure event on the pitch.

        Shows:
        - Pressed player (yellow square)
        - Active supporters (green, with speed labels)
        - Static teammates (red, faded)
        - Opponents (gray)
        - Ball position

        Args:
            pressure_result: PressureEventResult from analyzer
            frame_data: Tracking data for this frame
            save_path: Optional path to save figure
            show: Whether to display the figure

        Returns:
            matplotlib Figure object
        """
        # Create pitch
        pitch = Pitch(
            pitch_type='skillcorner',
            pitch_length=self.match_data.pitch_length,
            pitch_width=self.match_data.pitch_width,
            line_color=self.COLORS['pitch_lines'],
            pitch_color=self.COLORS['pitch_background'],
            linewidth=2
        )

        fig, ax = pitch.draw(figsize=(14, 9))

        # Get pressed player position
        pressed_id = pressure_result.pressed_player_id
        pressed_team = pressure_result.pressed_team_id

        # Get teammate details as dict for easy lookup
        teammate_dict = {tm['teammate_id']: tm for tm in pressure_result.teammate_details}

        # Plot all players
        for _, player in frame_data.iterrows():
            pid = int(player['player_id'])
            x, y = player['x'], player['y']
            team = int(player['team_id'])

            # Get velocity components if available
            vx = player.get('vx', 0) if pd.notna(player.get('vx')) else 0
            vy = player.get('vy', 0) if pd.notna(player.get('vy')) else 0

            # Determine marker style and color
            if pid == pressed_id:
                # Pressed player - yellow square
                ax.scatter(x, y, c=self.COLORS['pressed_player'], s=450,
                           edgecolors='black', linewidths=3, marker='s', zorder=10)
                ax.annotate('PRESSED', (x, y + 3), ha='center', fontsize=8,
                            color='yellow', fontweight='bold', zorder=11)
                # Add velocity arrow for pressed player
                if vx != 0 or vy != 0:
                    ax.quiver(x, y, vx, vy, color='yellow', scale=1, scale_units='xy',
                              width=0.008, headwidth=3, headlength=4, zorder=9)
            elif team == pressed_team and pid in teammate_dict:
                # Teammate - check if active supporter
                tm = teammate_dict[pid]
                if tm['is_active_supporter']:
                    # Active supporter - green
                    ax.scatter(x, y, c=self.COLORS['active_supporter'], s=350,
                               edgecolors='white', linewidths=2, marker='o', zorder=8)
                    # Add speed label
                    ax.annotate(f"{tm['speed_ms']:.1f}", (x, y - 3),
                                ha='center', fontsize=7, color='white', zorder=9)
                    # Add velocity arrow
                    if vx != 0 or vy != 0:
                        ax.quiver(x, y, vx, vy, color='white', scale=1, scale_units='xy',
                                  width=0.006, headwidth=3, headlength=4, zorder=7)
                elif tm['is_nearby']:
                    # Nearby but static - red
                    ax.scatter(x, y, c=self.COLORS['static_teammate'], s=350,
                               edgecolors='white', linewidths=2, marker='o',
                               alpha=0.7, zorder=6)
                    # Add velocity arrow (smaller for static)
                    if vx != 0 or vy != 0:
                        ax.quiver(x, y, vx, vy, color='white', scale=1, scale_units='xy',
                                  width=0.004, headwidth=3, headlength=4, alpha=0.5, zorder=5)
                else:
                    # Far teammate - dim light blue
                    ax.scatter(x, y, c=self.COLORS['home_team'], s=200,
                               edgecolors='white', linewidths=1, marker='o',
                               alpha=0.4, zorder=4)
                    # Add velocity arrow (faded for far teammates)
                    if vx != 0 or vy != 0:
                        ax.quiver(x, y, vx, vy, color='white', scale=1, scale_units='xy',
                                  width=0.003, headwidth=3, headlength=4, alpha=0.3, zorder=3)
            elif team != pressed_team:
                # Opponent
                ax.scatter(x, y, c=self.COLORS['opponent'], s=250,
                           edgecolors='white', linewidths=1, marker='o', zorder=5)
                # Add velocity arrow for opponents
                if vx != 0 or vy != 0:
                    ax.quiver(x, y, vx, vy, color='gray', scale=1, scale_units='xy',
                              width=0.004, headwidth=3, headlength=4, alpha=0.5, zorder=4)

            # Add jersey number (black for pressed player on yellow bg, white for others)
            number = str(int(player.get('number', 0))) if pd.notna(player.get('number')) else ""
            if number:
                num_color = 'black' if pid == pressed_id else 'white'
                ax.annotate(number, (x, y), ha='center', va='center',
                            fontsize=8, color=num_color, fontweight='bold', zorder=12)

        # Plot ball
        ball_x = frame_data['ball_x'].iloc[0] if 'ball_x' in frame_data.columns else np.nan
        ball_y = frame_data['ball_y'].iloc[0] if 'ball_y' in frame_data.columns else np.nan
        if pd.notna(ball_x):
            ax.scatter(ball_x, ball_y, c=self.COLORS['ball'], s=100,
                       edgecolors='black', linewidths=2, zorder=15)

        # Add team names and attack direction
        L = self.match_data.pitch_length / 2
        W = self.match_data.pitch_width / 2
        period = pressure_result.period

        home_name = self.match_data.home_team_name
        away_name = self.match_data.away_team_name
        home_dir = self.match_data.get_attack_direction(self.match_data.home_team_id, period)

        # Position labels at top corners of pitch with arrows showing attack direction
        # Team appears on the side where they DEFEND, arrow points toward where they ATTACK
        arrow_y = W + 5
        if home_dir == 'negative_x':
            # Home attacks left (negative x), defends on right
            # Away attacks right (positive x), defends on left
            ax.text(L * 0.85, arrow_y, f"← {home_name}", ha='center', fontsize=9,
                    color='white', fontweight='bold')
            ax.text(-L * 0.85, arrow_y, f"{away_name} →", ha='center', fontsize=9,
                    color='white', fontweight='bold')
        else:
            # Home attacks right (positive x), defends on left
            # Away attacks left (negative x), defends on right
            ax.text(-L * 0.85, arrow_y, f"{home_name} →", ha='center', fontsize=9,
                    color='white', fontweight='bold')
            ax.text(L * 0.85, arrow_y, f"← {away_name}", ha='center', fontsize=9,
                    color='white', fontweight='bold')

        # Add legend (use config thresholds, not hardcoded)
        vel_threshold = self.config.velocity_threshold_ms
        prox_threshold = self.config.proximity_threshold_m
        legend_elements = [
            mpatches.Patch(color=self.COLORS['pressed_player'], label='Pressed Player'),
            mpatches.Patch(color=self.COLORS['active_supporter'], label=f'Active Supporter (>{vel_threshold}m/s)'),
            mpatches.Patch(color=self.COLORS['static_teammate'], label=f'Static Teammate (<{prox_threshold:.0f}m)'),
            mpatches.Patch(color=self.COLORS['home_team'], alpha=0.4, label=f'Far Teammate (>{prox_threshold:.0f}m)'),
            mpatches.Patch(color=self.COLORS['opponent'], label='Opponent'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.8)

        # Build pressed player info with number and team
        pressed_row = frame_data[frame_data['player_id'] == pressed_id]
        pressed_number = ""
        if len(pressed_row) > 0 and 'number' in pressed_row.columns and pd.notna(pressed_row['number'].iloc[0]):
            pressed_number = f"#{int(pressed_row['number'].iloc[0])} "
        pressed_team_name = home_name if pressed_team == self.match_data.home_team_id else away_name
        pressed_info = f"{pressed_number}{pressure_result.pressed_player_name} ({pressed_team_name})"

        # Build match time string (format: MM:SS)
        match_time_min = frame_data['match_time_min'].iloc[0] if 'match_time_min' in frame_data.columns else None
        if match_time_min is not None and pd.notna(match_time_min):
            mins = int(match_time_min)
            secs = int((match_time_min - mins) * 60)
            time_str = f"{mins}:{secs:02d}"
        else:
            time_str = "N/A"

        # Get active support ratio (pre-calculated in asi_core)
        active = pressure_result.num_active_supporters
        nearby = pressure_result.num_teammates_nearby
        ratio_pct = int(round(pressure_result.active_support_ratio * 100))

        # Title with pressed player info and match time (3 lines)
        title = (f"Pressure Event: {pressure_result.pressure_subtype} | Pressed: {pressed_info}\n"
                 f"Active Supporters: {active}, Nearby Teammates: {nearby}, "
                 f"Active Support Ratio: ({active}/{nearby}) = {ratio_pct}%\n"
                 f"{time_str} ({period}H) | Frame {pressure_result.frame_start}")
        ax.set_title(title, fontsize=12, color='white', pad=10)

        fig.patch.set_facecolor('#1a1a1a')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor='#1a1a1a', edgecolor='none')

        if show:
            plt.show()

        return fig

    def plot_player_leaderboard(self, player_scores: pd.DataFrame,
                                  top_n: Optional[int] = 15,
                                  min_opportunities: int = 20,
                                  save_path: Optional[str] = None,
                                  show: bool = True) -> plt.Figure:
        """
        Create bar chart of player ASI scores.

        Args:
            player_scores: DataFrame from calculate_player_asi_scores()
            top_n: Number of top players to show
            min_opportunities: Minimum opportunities to include
            save_path: Optional path to save figure
            show: Whether to display

        Returns:
            matplotlib Figure
        """
        # Filter and sort
        df_full = player_scores[player_scores['opportunities'] >= min_opportunities].copy()
        df_full = df_full.sort_values('asi_score', ascending=True)

        # Calculate percentiles on FULL dataset (before top_n filter)
        p25 = df_full['asi_score'].quantile(0.25)
        p50 = df_full['asi_score'].quantile(0.50)
        p75 = df_full['asi_score'].quantile(0.75)

        # Apply top_n for display
        df = df_full.tail(top_n) if top_n is not None else df_full

        # Build display labels: "Player Name (TEAM - POS)"
        if 'team_name' in df.columns and 'player_role_acronym' in df.columns:
            df['display_label'] = df.apply(
                lambda r: f"{r['player_name']} ({r['team_name'][:3].upper()} - {r['player_role_acronym']})",
                axis=1
            )
        else:
            df['display_label'] = df['player_name']

        fig, ax = plt.subplots(figsize=(10, 8))

        # Create horizontal bar chart
        colors = plt.cm.RdYlGn(df['asi_score'])
        bars = ax.barh(df['display_label'], df['asi_score'], color=colors, edgecolor='white')

        # Add value labels with fraction format: 77.0% (214/278)
        has_matches_count = 'matches_count' in df.columns
        # Auto-hide n=... if all players have n=1 (single match)
        show_n = has_matches_count and df['matches_count'].max() > 1

        for i, bar in enumerate(bars):
            score = df['asi_score'].iloc[i]
            opps = int(df['opportunities'].iloc[i])
            active = int(df['active_support_count'].iloc[i])

            if show_n:
                n = int(df['matches_count'].iloc[i])
                label = f'{score:.1%} ({active}/{opps}), n={n}'
            else:
                label = f'{score:.1%} ({active}/{opps})'
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                    label, va='center', fontsize=9, color='white')

        # Styling
        ax.set_xlabel('ASI Score (Active Support Index)', fontsize=11)

        # Build title - include match name if single match
        match_info = ''
        if 'match_name' in df.columns:
            unique_matches = df['match_name'].dropna().unique()
            non_empty = [m for m in unique_matches if m]
            if len(non_empty) == 1:
                match_info = f' - {non_empty[0]}'

        if top_n is not None:
            title = f'Top {top_n} Players by ASI Score{match_info}\n(min {min_opportunities} opportunities)'
        else:
            title = f'All Players by ASI Score{match_info}\n(min {min_opportunities} opportunities)'
        ax.set_title(title, fontsize=13, pad=10)
        ax.set_xlim(0, 1.0)

        # Add percentile threshold lines (calculated on full dataset above)
        ax.axvline(x=p25, color='red', linestyle='--', alpha=0.5, label=f'25th pctl ({p25:.0%})')
        ax.axvline(x=p50, color='yellow', linestyle='--', alpha=0.5, label=f'50th pctl ({p50:.0%})')
        ax.axvline(x=p75, color='green', linestyle='--', alpha=0.5, label=f'75th pctl ({p75:.0%})')

        ax.legend(loc='lower right', fontsize=8)

        # Grid
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, alpha=0.3)

        fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#1a1a1a')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor='#1a1a1a', edgecolor='none')

        if show:
            plt.show()

        return fig

    def plot_team_comparison(self, team_stats: dict,
                               save_path: Optional[str] = None,
                               show: bool = True) -> plt.Figure:
        """
        Compare ASI metrics between teams.

        Args:
            team_stats: Dict from calculate_team_asi_scores()
            save_path: Optional path to save
            show: Whether to display

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))

        teams = list(team_stats.values())
        names = [t['team_name'] for t in teams]
        colors = [self.COLORS['home_team'], self.COLORS['away_team']]

        # Plot 1: Team ASI (1 - static rate)
        asi_scores = [t['team_asi'] for t in teams]
        axes[0].bar(names, asi_scores, color=colors, edgecolor='white')
        axes[0].set_ylabel('Team ASI', fontsize=10)
        axes[0].set_title('Team ASI Score', fontsize=11)
        axes[0].set_ylim(0.8, 1.0)
        for i, v in enumerate(asi_scores):
            axes[0].text(i, v - 0.01, f'{v:.1%}', ha='center', va='top', fontsize=10, color='white')

        # Plot 2: Avg Active Supporters
        avg_active = [t['avg_active_supporters'] for t in teams]
        axes[1].bar(names, avg_active, color=colors, edgecolor='white')
        axes[1].set_ylabel('Avg Active Supporters', fontsize=10)
        axes[1].set_title('Average Active Supporters', fontsize=11)
        for i, v in enumerate(avg_active):
            axes[1].text(i, v - 0.05, f'{v:.2f}', ha='center', va='top', fontsize=10, color='white')

        # Plot 3: Static Rate
        static_rates = [t['static_rate'] for t in teams]
        axes[2].bar(names, static_rates, color=colors, edgecolor='white')
        axes[2].set_ylabel('Static Rate', fontsize=10)
        axes[2].set_title('Static Rate (Lower = Better)', fontsize=11)
        for i, v in enumerate(static_rates):
            axes[2].text(i, v - 0.005, f'{v:.1%}', ha='center', va='top', fontsize=10, color='white')

        for ax in axes:
            ax.set_facecolor('#1a1a1a')
            ax.tick_params(colors='white')

        fig.patch.set_facecolor('#1a1a1a')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight',
                        facecolor='#1a1a1a', edgecolor='none')

        if show:
            plt.show()

        return fig


def plot_multi_match_team_comparison(team_summary: 'pd.DataFrame',
                                      save_path: str = None,
                                      show: bool = True) -> 'plt.Figure':
    """
    Create bar chart comparing all teams across multiple matches.

    Args:
        team_summary: DataFrame from get_team_stats_all_matches()
        save_path: Optional path to save figure
        show: Whether to display

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    teams = team_summary['team_name'].tolist()
    n_teams = len(teams)

    # Color palette for multiple teams
    colors = plt.cm.tab10(np.linspace(0, 1, n_teams))

    # Truncate team names for display
    short_names = [t[:15] + '...' if len(t) > 15 else t for t in teams]

    # Plot 1: Overall Team ASI
    asi_scores = team_summary['overall_team_asi'].tolist()
    bars1 = axes[0].bar(short_names, asi_scores, color=colors, edgecolor='white')
    axes[0].set_ylabel('Team ASI', fontsize=10)
    axes[0].set_title('Overall Team ASI\n(All Matches)', fontsize=11)
    axes[0].set_ylim(0.9, 1.0)
    for i, v in enumerate(asi_scores):
        axes[0].text(i, v - 0.005, f'{v:.1%}', ha='center', va='top', fontsize=8, color='white')
    axes[0].tick_params(axis='x', rotation=45)

    # Plot 2: Avg Active Supporters
    avg_active = team_summary['avg_active_supporters'].tolist()
    bars2 = axes[1].bar(short_names, avg_active, color=colors, edgecolor='white')
    axes[1].set_ylabel('Avg Active Supporters', fontsize=10)
    axes[1].set_title('Average Active Supporters\n(Per Pressure Event)', fontsize=11)
    for i, v in enumerate(avg_active):
        axes[1].text(i, v - 0.1, f'{v:.2f}', ha='center', va='top', fontsize=8, color='white')
    axes[1].tick_params(axis='x', rotation=45)

    # Plot 3: Matches Played (context)
    matches = team_summary['matches_played'].tolist()
    bars3 = axes[2].bar(short_names, matches, color=colors, edgecolor='white')
    axes[2].set_ylabel('Matches Played', fontsize=10)
    axes[2].set_title('Sample Size\n(Matches in Dataset)', fontsize=11)
    for i, v in enumerate(matches):
        axes[2].text(i, v - 0.1, str(v), ha='center', va='top', fontsize=8, color='white')
    axes[2].tick_params(axis='x', rotation=45)

    for ax in axes:
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white')

    fig.patch.set_facecolor('#1a1a1a')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='#1a1a1a', edgecolor='none')

    if show:
        plt.show()

    return fig
