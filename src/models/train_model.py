import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os


def train_model(X, y, test_size=0.2, random_state=42):
    """
    Train a linear regression model to predict driver championship points.

    Args:
        X: Feature matrix
        y: Target variable (total points)
        test_size:  Proportion of data for testing
        random_state: Random seed

    Returns:
        Trained model, predictions, metrics
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")

    metrics = {
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "test_mae": test_mae,
        "cv_r2_mean": cv_scores.mean(),
        "cv_r2_std": cv_scores.std(),
    }

    # Feature importance
    feature_importance = pd.DataFrame(
        {"feature": X.columns, "coefficient": model.coef_}
    ).sort_values("coefficient", ascending=False)

    return (
        model,
        (X_train, X_test, y_train, y_test, y_pred_train, y_pred_test),
        metrics,
        feature_importance,
    )


def save_model(model, filepath="models/linear_regression_model.pkl"):
    """Save trained model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"✓ Model saved to {filepath}")


def load_model(filepath="models/linear_regression_model.pkl"):
    """Load trained model from disk."""
    model = joblib.load(filepath)
    print(f"✓ Model loaded from {filepath}")
    return model
