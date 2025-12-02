import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def plot_indicator_trend(df: pd.DataFrame, indicator: str, title: str) -> go.Figure:
    """
    Generates a line chart for a given indicator over time.
    """
    # Ensure sorting by year
    df_sorted = df.sort_values(by="year")
    
    fig = px.line(df_sorted, x="year", y=indicator, title=title, markers=True)
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title=indicator, 
        hovermode="x unified",
        template="plotly_dark"
    )
    return fig


def plot_predictions_vs_actuals(
    combined_X: pd.DataFrame, 
    combined_y_actual: pd.Series, 
    combined_y_pred: pd.Series, 
    title: str, 
    target_column: str
) -> go.Figure:
    """
    Generates a plot comparing actual vs. predicted values.
    Accepts combined dataframes (Train + Test + Future).
    """
    results_df = pd.DataFrame({
        "Year": combined_X['year'],
        "Actual": combined_y_actual.values,
        "Predicted": combined_y_pred.values,
    }).sort_values(by="Year")

    fig = go.Figure()
    
    # Real data (blue)
    fig.add_trace(
        go.Scatter(
            x=results_df["Year"],
            y=results_df["Actual"],
            mode="lines+markers",
            name="Actual Data",
            line=dict(color="#2E86C1", width=3),
            marker=dict(size=8)
        )
    )
    
    # Predicted line (red)
    fig.add_trace(
        go.Scatter(
            x=results_df["Year"],
            y=results_df["Predicted"],
            mode="lines+markers",
            name="Model Prediction",
            line=dict(color="#E74C3C", dash="dash", width=3),
            marker=dict(size=8, symbol="diamond")
        )
    )

    future_start_idx = results_df["Actual"].isna().idxmax() if results_df["Actual"].isna().any() else None
    
    if future_start_idx is not None:
        future_year_start = results_df.loc[future_start_idx, "Year"]
        # Vertical line do separate prediction
        fig.add_vline(
            x=future_year_start - 0.5, 
            line_width=1, 
            line_dash="dot", 
            line_color="gray",
            annotation_text="Forecast Start", 
            annotation_position="top right"
        )

    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title=target_column,
        hovermode="x unified",
        template="plotly_dark",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig


def plot_feature_importance(feature_importance: pd.Series, title: str) -> go.Figure:
    """
    Generates a horizontal bar chart visualization of feature importance scores.
    """
    if feature_importance is None or len(feature_importance) == 0:
        fig = go.Figure()
        fig.update_layout(title="No feature importance available")
        return fig

    if isinstance(feature_importance, pd.DataFrame):
        if feature_importance.shape[1] >= 2:
            feature_importance = pd.Series(
                data=feature_importance.iloc[:, 1].values, 
                index=feature_importance.iloc[:, 0].values
            )

    fi = feature_importance.copy()
    fi = fi.sort_values(ascending=True)
    
    fig = px.bar(
        x=fi.values,
        y=fi.index,
        orientation='h',
        labels={'x': 'Relative Importance (%)', 'y': 'Feature'},
        title=title,
        text_auto='.1f'
    )
    
    fig.update_traces(textposition='outside', marker_color='#3498DB')
    
    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20), 
        height=max(300, 40 * len(fi)),
        template="plotly_dark"
    )
    return fig