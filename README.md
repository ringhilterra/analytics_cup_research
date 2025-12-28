# SkillCorner X PySport Analytics Cup - Ryan Inghiltera

---

## Absract - The Active Support Index (ASI): Quantifying Off-Ball Movement During Pressure Events

**Author: Ryan Inghilterra**

---

## Introduction

When a ball carrier is pressured by an opponent, how actively do nearby teammates move to provide passing options? This question addresses a gap in football analytics: traditional metrics (xG, xA, pass completion) focus on ball events, leaving off-ball movement largely unquantified. This research introduces the **Active Support Index (ASI)**, a metric framework that measures teammate support during pressure moments using Skillcorner tracking, dynamic_events, and player physical aggregate data. ASI quantifies how often nearby teammates are actively moving versus standing static when a pressured teammate needs support.

## Methods

We analyzed 10 A-League 2024/25 matches containing 7,063 pressure events using SkillCorner's 10fps tracking data. Player velocities were calculated using a Savitzky-Golay filter (window=7 frames, polynomial degree=2) applied to positional data.

**Metric Definitions:**

| Level | Metric | Formula |
|-------|--------|---------|
| Per Event | Active Support Ratio | Active Supporters / Nearby Teammates |
| Per Player | Player ASI | Active Support Count / Support Opportunities |
| Per Team | Team ASI | 1 − Static Rate |

**Key Definitions:**
- **Active Supporter**: Teammate within 35m of the pressed player AND moving >2 m/s
- **Static Rate**: Proportion of pressure events with zero active supporters
- **Proximity threshold (35m)**: Maximum realistic passing range under pressure
- **Velocity threshold (2 m/s)**: Boundary separating walking from jogging/running

For validation, we compared player ASI scores (calculated from 10 tracking matches) against season-level physical aggregates from 175 A-League matches, specifically meters per minute during team possession (`total_metersperminute_full_tip`).

## Results

**Figure 1** illustrates a low-ASI pressure event where only 1 of 9 nearby teammates (11%) was actively moving while the ball carrier was pressed—8 teammates were static.

![Figure 1: Low ASI pressure event showing 8 of 9 teammates static](figs/fig1_low_asi_pressure_event.png)

*Figure 1: Pressure event with Active Support Ratio = 0.11. Only player #17 (2.2 m/s) provides active support while 8 nearby teammates remain static (<2 m/s).*

**External Validation:** ASI correlates strongly with season-level physical output (Pearson r = 0.74, p = 4.47e-30, n = 167 players). Players in the top ASI quartile cover 153.2 m/min during possession versus 128.8 m/min for the bottom quartile—a 19% difference in work rate (**Figure 2**).

![Figure 2: ASI vs physical output correlation](figs/fig2_asi_physical_validation.png)

*Figure 2: Player ASI scores versus season-level meters per minute during possession. Strong positive correlation (r = 0.74) validates that ASI captures genuine physical behavior.*

**Position Validation:** ASI aligns with expected positional demands. Midfielders average 59.4% ASI versus 45.2% for defenders (Mann-Whitney U = 4092, p = 4.00e-14).

**Team Differentiation:** Team ASI ranges from Perth Glory (98.3%) to Macarthur FC (91.5%), a 7-point spread indicating ASI can differentiate team off-ball movement cultures.

**Fatigue Analysis:** 58% of players (n=151) showed declining ASI in the second half, with average support ratio dropping from 53.7% to 53.1% (not statistically significant, p = 0.46).

## Conclusion

The Active Support Index provides a quantitative framework for measuring off-ball support during pressure events. The strong correlation with independent physical aggregate data (r = 0.74 across 175 matches) validates that ASI captures real physical behavior rather than arbitrary thresholds. Applications include identifying players who consistently provide active support under pressure, diagnosing static tendencies for coaching intervention, and comparing team tactical styles.


---

## Installation & Usage

### Option 1: Run in Google Colab

No setup required. Simply click the badge below to open and run the notebook directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ringhilterra/analytics_cup_research/blob/main/submission.ipynb)

### Option 2: Run Locally

Tested on Python 3.12 (macOS)

1. Clone the repository:
   ```bash
   git clone https://github.com/ringhilterra/analytics_cup_research.git
   cd analytics_cup_research
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:
   ```bash
   jupyter notebook submission.ipynb
   ```

---