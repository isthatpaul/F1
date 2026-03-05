import pandas as pd
import numpy as np

FEATURE_COLS = [
    "AvgPoints",
    "AvgFinish",
    "AvgGrid",
    "PodiumRate",
    "WinRate",
    "DNFRate",
    "AvgPositionChange",
    "RollingAvgPoints",
    "RollingPodiumRate",
    "PrevYearPoints",
    "PrevYearPodiums",
]


def build_driver_features(season_df):
    """
    Build comprehensive driver features for modeling.

    Features include:
    - Total points
    - Average points per race
    - Podium rate
    - DNF rate
    - Average starting position
    - Average finishing position
    - Consistency (std of points)
    """

    driver_stats = []

    for driver in season_df["DriverFull"].unique():
        driver_data = season_df[season_df["DriverFull"] == driver].copy()

        # Basic stats
        total_races = len(driver_data)
        total_points = driver_data["Points"].sum()
        avg_points = driver_data["Points"].mean()
        std_points = driver_data["Points"].std()

        # Position stats
        avg_finish = driver_data["Position"].mean()
        median_finish = driver_data["Position"].median()
        avg_grid = driver_data["GridPosition"].mean()

        # Podiums and wins
        podiums = (driver_data["Position"] <= 3).sum()
        wins = (driver_data["Position"] == 1).sum()
        podium_rate = podiums / total_races if total_races > 0 else 0
        win_rate = wins / total_races if total_races > 0 else 0

        # DNFs
        dnfs = driver_data["Position"].isna().sum()
        dnf_rate = dnfs / total_races if total_races > 0 else 0

        # Positions gained/lost
        driver_data["PositionChange"] = (
            driver_data["GridPosition"] - driver_data["Position"]
        )
        avg_position_change = driver_data["PositionChange"].mean()

        # Team (most common team for the driver in this season)
        team = (
            driver_data["Team"].mode()[0]
            if len(driver_data["Team"].mode()) > 0
            else "Unknown"
        )

        # Year
        year = driver_data["Year"].iloc[0]

        driver_stats.append(
            {
                "Year": year,
                "Driver": driver,
                "Team": team,
                "TotalRaces": total_races,
                "TotalPoints": total_points,
                "AvgPoints": avg_points,
                "StdPoints": std_points,
                "AvgFinish": avg_finish,
                "MedianFinish": median_finish,
                "AvgGrid": avg_grid,
                "Podiums": podiums,
                "Wins": wins,
                "PodiumRate": podium_rate,
                "WinRate": win_rate,
                "DNFs": dnfs,
                "DNFRate": dnf_rate,
                "AvgPositionChange": avg_position_change,
            }
        )

    features_df = pd.DataFrame(driver_stats)

    # Sort by total points
    features_df = features_df.sort_values("TotalPoints", ascending=False).reset_index(
        drop=True
    )

    return features_df


def add_rolling_features(features_df, window=3):
    """
    Add rolling/temporal features (e.g., form from previous seasons).

    Args:
        features_df: DataFrame with yearly driver features
        window: Number of previous seasons to consider

    Returns:
        DataFrame with rolling features
    """
    features_df = features_df.sort_values(["Driver", "Year"])

    # Rolling averages
    features_df["RollingAvgPoints"] = features_df.groupby("Driver")[
        "TotalPoints"
    ].transform(lambda x: x.rolling(window=window, min_periods=1).mean())

    features_df["RollingPodiumRate"] = features_df.groupby("Driver")[
        "PodiumRate"
    ].transform(lambda x: x.rolling(window=window, min_periods=1).mean())

    # Previous year performance
    features_df["PrevYearPoints"] = features_df.groupby("Driver")["TotalPoints"].shift(
        1
    )
    features_df["PrevYearPodiums"] = features_df.groupby("Driver")["Podiums"].shift(1)

    # Fill NaN for drivers in their first year
    features_df["PrevYearPoints"] = features_df["PrevYearPoints"].fillna(0)
    features_df["PrevYearPodiums"] = features_df["PrevYearPodiums"].fillna(0)

    return features_df


def prepare_training_data(features_df):
    """
    Prepare features for machine learning model.

    Returns:
        X (features), y (target - total points)
    """
    # Target variable
    y = features_df["TotalPoints"]

    # Feature columns
    X = features_df[FEATURE_COLS]

    return X, y
