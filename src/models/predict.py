import pandas as pd
import numpy as np
from src.features.build_features import FEATURE_COLS


def predict_2026_champion(model, features_2025_df):
    """
    Predict 2026 championship results based on 2025 performance.

    Args:
        model: Trained linear regression model
        features_2025_df: Feature DataFrame for 2025 drivers

    Returns:
        DataFrame with predictions
    """
    X_2026 = features_2025_df[FEATURE_COLS]

    # Predict
    predicted_points = model.predict(X_2026)

    # Create results DataFrame
    predictions_df = pd.DataFrame(
        {
            "Driver": features_2025_df["Driver"],
            "Team": features_2025_df["Team"],
            "Predicted2026Points": predicted_points,
            "PrevYearPoints": features_2025_df["TotalPoints"],
        }
    )

    # Sort by predicted points
    predictions_df = predictions_df.sort_values(
        "Predicted2026Points", ascending=False
    ).reset_index(drop=True)
    predictions_df["PredictedPosition"] = predictions_df.index + 1

    return predictions_df


def calculate_win_probability(predictions_df):
    """
    Calculate probability of winning championship based on predicted points.
    Using softmax to convert points to probabilities.
    """
    points = predictions_df["Predicted2026Points"].values

    # Softmax
    exp_points = np.exp(points / 100)  # Scale down for numerical stability
    probabilities = exp_points / exp_points.sum()

    predictions_df["WinProbability"] = probabilities * 100  # As percentage

    return predictions_df
