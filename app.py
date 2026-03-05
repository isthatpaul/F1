import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="F1 2026 Championship Prediction",
    page_icon="🏎️",
    layout="wide",
)


# Load data
@st.cache_data
def load_predictions():
    return pd.read_csv("data/predictions/2026_championship_predictions.csv")


@st.cache_data
def load_feature_importance():
    return pd.read_csv("models/feature_importance.csv")


@st.cache_data
def load_model_metrics():
    return pd.read_csv("models/model_metrics.csv")


predictions = load_predictions()
features = load_feature_importance()
metrics = load_model_metrics().iloc[0]

# Sidebar
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/3/33/F1.svg", width=120
)
st.sidebar.title("🏎️ F1 2026 Predictor")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏆 Championship Predictions", "📊 Model Performance", "🔍 Driver Deep Dive"],
)

st.sidebar.markdown("---")
st.sidebar.caption("Built with FastF1, scikit-learn & Streamlit")
# Page 1: Championship Predictions

if page == "🏆 Championship Predictions":
    st.title("🏆 2026 F1 Drivers' Championship Prediction")
    st.markdown(
        "Predicted standings based on historical performance data (2023–2025) and linear regression."
    )

    # Top 3 podium
    col1, col2, col3 = st.columns(3)
    top3 = predictions.head(3)

    with col1:
        st.metric(
            label="🥇 1st Place",
            value=top3.iloc[0]["Driver"],
            delta=f'{top3.iloc[0]["Predicted2026Points"]:.0f} pts',
        )
    with col2:
        st.metric(
            label="🥈 2nd Place",
            value=top3.iloc[1]["Driver"],
            delta=f'{top3.iloc[1]["Predicted2026Points"]:.0f} pts',
        )
    with col3:
        st.metric(
            label="🥉 3rd Place",
            value=top3.iloc[2]["Driver"],
            delta=f'{top3.iloc[2]["Predicted2026Points"]:.0f} pts',
        )

    st.markdown("---")

    # Top N slider
    top_n = st.slider(
        "Show top N drivers", min_value=5, max_value=len(predictions), value=10
    )
    top_df = predictions.head(top_n)

    # Predicted points bar chart
    fig_points = px.bar(
        top_df,
        x="Driver",
        y="Predicted2026Points",
        color="Team",
        title=f"Predicted Championship Points — Top {top_n}",
        labels={"Predicted2026Points": "Predicted Points"},
        text=top_df["Predicted2026Points"].round(0).astype(int),
    )
    fig_points.update_traces(textposition="outside")
    fig_points.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig_points, use_container_width=True)

    # Win probability pie chart
    fig_prob = px.pie(
        top_df,
        values="WinProbability",
        names="Driver",
        title=f"Win Probability Distribution — Top {top_n}",
        hole=0.4,
    )
    fig_prob.update_traces(textinfo="label+percent")
    st.plotly_chart(fig_prob, use_container_width=True)

    # Full standings table
    st.subheader("📋 Full Predicted Standings")
    display_df = predictions.copy()
    display_df["Predicted2026Points"] = display_df["Predicted2026Points"].round(1)
    display_df["WinProbability"] = (display_df["WinProbability"]).round(2).astype(
        str
    ) + "%"
    display_df = display_df.rename(
        columns={
            "Predicted2026Points": "Predicted Points",
            "PrevYearPoints": "2025 Points",
            "PredictedPosition": "Position",
            "WinProbability": "Win Probability",
        }
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# Page 2: Model Performance

elif page == "📊 Model Performance":
    st.title("📊 Model Performance")
    st.markdown("Linear Regression model trained on 2023–2025 F1 season data.")

    # Metrics cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Train R²", f'{metrics["train_r2"]:.4f}')
    with col2:
        st.metric("Test R²", f'{metrics["test_r2"]:.4f}')
    with col3:
        st.metric("Test RMSE", f'{metrics["test_rmse"]:.2f} pts')
    with col4:
        st.metric("Test MAE", f'{metrics["test_mae"]:.2f} pts')

    st.markdown("---")

    # Cross-validation
    st.subheader("🔄 Cross-Validation")
    cv_col1, cv_col2 = st.columns(2)
    with cv_col1:
        st.metric("CV R² (mean)", f'{metrics["cv_r2_mean"]:.4f}')
    with cv_col2:
        st.metric("CV R² (std)", f'± {metrics["cv_r2_std"]:.4f}')

    st.markdown("---")

    # Feature importance
    st.subheader("🎯 Feature Importance (Linear Regression Coefficients)")

    features_sorted = features.sort_values("coefficient", ascending=True)
    features_sorted["abs_coeff"] = features_sorted["coefficient"].abs()
    features_sorted["direction"] = features_sorted["coefficient"].apply(
        lambda x: "Positive" if x >= 0 else "Negative"
    )

    fig_feat = px.bar(
        features_sorted,
        x="coefficient",
        y="feature",
        orientation="h",
        color="direction",
        color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c"},
        title="Feature Coefficients",
        labels={"coefficient": "Coefficient Value", "feature": "Feature"},
    )
    fig_feat.update_layout(height=500)
    st.plotly_chart(fig_feat, use_container_width=True)

    # Metrics table
    st.subheader("📋 Raw Metrics")
    metrics_display = pd.DataFrame(
        {
            "Metric": [
                "Train RMSE",
                "Test RMSE",
                "Train R²",
                "Test R²",
                "Test MAE",
                "CV R² Mean",
                "CV R² Std",
            ],
            "Value": [
                f'{metrics["train_rmse"]:.4f}',
                f'{metrics["test_rmse"]:.4f}',
                f'{metrics["train_r2"]:.4f}',
                f'{metrics["test_r2"]:.4f}',
                f'{metrics["test_mae"]:.4f}',
                f'{metrics["cv_r2_mean"]:.4f}',
                f'{metrics["cv_r2_std"]:.4f}',
            ],
        }
    )
    st.dataframe(metrics_display, use_container_width=True, hide_index=True)

# Page 3: Driver Deep Dive
elif page == "🔍 Driver Deep Dive":
    st.title("🔍 Driver Deep Dive")

    selected_driver = st.selectbox("Select a driver", predictions["Driver"].tolist())
    driver_data = predictions[predictions["Driver"] == selected_driver].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Predicted Position", f'P{int(driver_data["PredictedPosition"])}')
    with col2:
        st.metric("Predicted Points", f'{driver_data["Predicted2026Points"]:.0f}')
    with col3:
        st.metric("2025 Points", f'{driver_data["PrevYearPoints"]:.0f}')
    with col4:
        st.metric("Win Probability", f'{driver_data["WinProbability"]:.1f}%')

    st.markdown("---")

    # Points comparison: 2025 vs 2026 predicted
    st.subheader(f"📈 {selected_driver} — 2025 Actual vs 2026 Predicted")
    compare_df = pd.DataFrame(
        {
            "Season": ["2025 (Actual)", "2026 (Predicted)"],
            "Points": [
                driver_data["PrevYearPoints"],
                driver_data["Predicted2026Points"],
            ],
        }
    )
    fig_compare = px.bar(
        compare_df,
        x="Season",
        y="Points",
        color="Season",
        text=compare_df["Points"].round(0).astype(int),
        title=f"{selected_driver} — Points Comparison",
    )
    fig_compare.update_traces(textposition="outside")
    fig_compare.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_compare, use_container_width=True)

    # Where they stand among teammates
    team = driver_data["Team"]
    teammates = predictions[predictions["Team"] == team]
    st.subheader(f"👥 {team} Teammate Comparison")
    fig_team = px.bar(
        teammates,
        x="Driver",
        y="Predicted2026Points",
        color="Driver",
        text=teammates["Predicted2026Points"].round(0).astype(int),
        title=f"{team} — Predicted Points",
        labels={"Predicted2026Points": "Predicted Points"},
    )
    fig_team.update_traces(textposition="outside")
    fig_team.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_team, use_container_width=True)
