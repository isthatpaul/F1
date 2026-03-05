import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def plot_model_performance(y_train, y_pred_train, y_test, y_pred_test, metrics):
    """
    Visualize model performance with actual vs predicted plots.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Training set
    axes[0].scatter(y_train, y_pred_train, alpha=0.6, edgecolors="k", linewidth=0.5)
    axes[0].plot(
        [y_train.min(), y_train.max()],
        [y_train.min(), y_train.max()],
        "r--",
        lw=2,
        label="Perfect Prediction",
    )
    axes[0].set_xlabel("Actual Points", fontsize=12)
    axes[0].set_ylabel("Predicted Points", fontsize=12)
    axes[0].set_title(
        f'Training Set\nR² = {metrics["train_r2"]:.3f}, RMSE = {metrics["train_rmse"]:.2f}',
        fontsize=12,
        fontweight="bold",
    )
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Test set
    axes[1].scatter(
        y_test, y_pred_test, alpha=0.6, color="orange", edgecolors="k", linewidth=0.5
    )
    axes[1].plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--",
        lw=2,
        label="Perfect Prediction",
    )
    axes[1].set_xlabel("Actual Points", fontsize=12)
    axes[1].set_ylabel("Predicted Points", fontsize=12)
    axes[1].set_title(
        f'Test Set\nR² = {metrics["test_r2"]:.3f}, RMSE = {metrics["test_rmse"]:.2f}',
        fontsize=12,
        fontweight="bold",
    )
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_residuals(y_test, y_pred_test):
    """
    Plot residuals to check for patterns.
    """
    residuals = y_test - y_pred_test

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residual plot
    axes[0].scatter(y_pred_test, residuals, alpha=0.6, edgecolors="k", linewidth=0.5)
    axes[0].axhline(y=0, color="r", linestyle="--", lw=2)
    axes[0].set_xlabel("Predicted Points", fontsize=12)
    axes[0].set_ylabel("Residuals", fontsize=12)
    axes[0].set_title("Residual Plot", fontsize=12, fontweight="bold")
    axes[0].grid(alpha=0.3)

    # Residual distribution
    axes[1].hist(residuals, bins=20, edgecolor="black", alpha=0.7)
    axes[1].axvline(x=0, color="r", linestyle="--", lw=2)
    axes[1].set_xlabel("Residuals", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_title("Residual Distribution", fontsize=12, fontweight="bold")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_feature_importance(feature_importance, top_n=10):
    """
    Visualize feature importance from linear regression coefficients.
    """
    top_features = feature_importance.head(top_n)

    plt.figure(figsize=(10, 6))
    colors = ["green" if x > 0 else "red" for x in top_features["coefficient"]]
    plt.barh(
        top_features["feature"],
        top_features["coefficient"],
        color=colors,
        alpha=0.7,
        edgecolor="black",
    )
    plt.xlabel("Coefficient Value", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.title(
        f"Top {top_n} Feature Importance (Linear Regression Coefficients)",
        fontsize=14,
        fontweight="bold",
    )
    plt.axvline(x=0, color="black", linestyle="--", linewidth=1)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_predictions_2026(predictions_df, top_n=10):
    """
    Visualize 2026 championship predictions.
    """
    top_drivers = predictions_df.head(top_n)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Predicted points
    axes[0].barh(
        top_drivers["Driver"],
        top_drivers["Predicted2026Points"],
        color="steelblue",
        alpha=0.8,
        edgecolor="black",
    )
    axes[0].set_xlabel("Predicted Points", fontsize=12)
    axes[0].set_ylabel("Driver", fontsize=12)
    axes[0].set_title(
        "2026 F1 Championship - Predicted Points (Top 10)",
        fontsize=14,
        fontweight="bold",
    )
    axes[0].invert_yaxis()
    axes[0].grid(axis="x", alpha=0.3)

    # Add values on bars
    for i, (driver, points) in enumerate(
        zip(top_drivers["Driver"], top_drivers["Predicted2026Points"])
    ):
        axes[0].text(points + 5, i, f"{points:.1f}", va="center", fontsize=10)

    # Win probability
    axes[1].barh(
        top_drivers["Driver"],
        top_drivers["WinProbability"],
        color="coral",
        alpha=0.8,
        edgecolor="black",
    )
    axes[1].set_xlabel("Win Probability (%)", fontsize=12)
    axes[1].set_ylabel("Driver", fontsize=12)
    axes[1].set_title(
        "2026 F1 Championship - Win Probability (Top 10)",
        fontsize=14,
        fontweight="bold",
    )
    axes[1].invert_yaxis()
    axes[1].grid(axis="x", alpha=0.3)

    # Add values on bars
    for i, (driver, prob) in enumerate(
        zip(top_drivers["Driver"], top_drivers["WinProbability"])
    ):
        axes[1].text(prob + 0.5, i, f"{prob:.2f}%", va="center", fontsize=10)

    plt.tight_layout()
    plt.show()


def plot_comparison_2025_vs_2026(predictions_df, top_n=10):
    """
    Compare 2025 actual vs 2026 predicted performance.
    """
    top_drivers = predictions_df.head(top_n)

    x = np.arange(len(top_drivers))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(
        x - width / 2,
        top_drivers["PrevYearPoints"],
        width,
        label="2025 Actual",
        color="lightblue",
        edgecolor="black",
        alpha=0.8,
    )
    plt.bar(
        x + width / 2,
        top_drivers["Predicted2026Points"],
        width,
        label="2026 Predicted",
        color="salmon",
        edgecolor="black",
        alpha=0.8,
    )

    plt.xlabel("Driver", fontsize=12)
    plt.ylabel("Championship Points", fontsize=12)
    plt.title(
        "2025 Actual vs 2026 Predicted Championship Points",
        fontsize=14,
        fontweight="bold",
    )
    plt.xticks(x, top_drivers["Driver"], rotation=45, ha="right")
    plt.legend(fontsize=11)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_interactive_predictions(predictions_df):
    """
    Create interactive Plotly visualization of predictions.
    """
    top_20 = predictions_df.head(20)

    fig = go.Figure()

    # Add 2025 actual points
    fig.add_trace(
        go.Bar(
            x=top_20["Driver"],
            y=top_20["PrevYearPoints"],
            name="2025 Actual",
            marker_color="lightblue",
            text=top_20["PrevYearPoints"].round(1),
            textposition="outside",
        )
    )

    # Add 2026 predicted points
    fig.add_trace(
        go.Bar(
            x=top_20["Driver"],
            y=top_20["Predicted2026Points"],
            name="2026 Predicted",
            marker_color="salmon",
            text=top_20["Predicted2026Points"].round(1),
            textposition="outside",
        )
    )

    fig.update_layout(
        title="F1 Championship Points:  2025 Actual vs 2026 Predicted (Top 20)",
        xaxis_title="Driver",
        yaxis_title="Points",
        barmode="group",
        height=600,
        hovermode="x unified",
        template="plotly_white",
    )

    fig.show()

    # Win probability pie chart
    top_5 = predictions_df.head(5)

    fig2 = go.Figure(
        data=[
            go.Pie(
                labels=top_5["Driver"],
                values=top_5["WinProbability"],
                hole=0.3,
                marker=dict(colors=px.colors.qualitative.Set3),
            )
        ]
    )

    fig2.update_layout(
        title="2026 Championship Win Probability (Top 5 Drivers)", height=500
    )

    fig2.show()


def plot_historical_performance(features_df, drivers_to_plot):
    """
    Plot historical performance trends for selected drivers.
    """
    plt.figure(figsize=(12, 6))

    for driver in drivers_to_plot:
        driver_data = features_df[features_df["Driver"] == driver].sort_values("Year")
        plt.plot(
            driver_data["Year"],
            driver_data["TotalPoints"],
            marker="o",
            linewidth=2,
            markersize=8,
            label=driver,
        )

    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Total Championship Points", fontsize=12)
    plt.title("Historical Performance Trends", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def create_summary_report(predictions_df, metrics):
    """
    Create a text summary report.
    """
    print("=" * 70)
    print(" " * 20 + "2026 F1 CHAMPIONSHIP PREDICTION")
    print("=" * 70)
    print("\n📊 MODEL PERFORMANCE:")
    print(f"  • Test R² Score: {metrics['test_r2']:.4f}")
    print(f"  • Test RMSE: {metrics['test_rmse']:.2f} points")
    print(f"  • Test MAE: {metrics['test_mae']:.2f} points")
    print(
        f"  • Cross-Validation R² (mean): {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}"
    )

    print("\n" + "=" * 70)
    print("🏆 TOP 5 CHAMPIONSHIP CONTENDERS:")
    print("=" * 70)

    top_5 = predictions_df.head(5)
    for idx, row in top_5.iterrows():
        print(f"\n{row['PredictedPosition']}.  {row['Driver']} ({row['Team']})")
        print(f"   Predicted Points: {row['Predicted2026Points']:.1f}")
        print(f"   Win Probability: {row['WinProbability']:.2f}%")
        print(f"   2025 Points: {row['PrevYearPoints']:.1f}")

    print("\n" + "=" * 70)
    print(f"🎯 PREDICTED 2026 CHAMPION: {top_5.iloc[0]['Driver']}")
    print("=" * 70)
