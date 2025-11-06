import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_indicator_trend(df: pd.DataFrame, indicator: str, title: str) -> go.Figure:
    """
    Generates a line chart for a given indicator over time.

    Args:
        df (pd.DataFrame): The data for a single country.
        indicator (str): The indicator to plot.
        title (str): The title for the plot.

    Returns:
        go.Figure: A Plotly figure object.
    """
    fig = px.line(df, x="year", y=indicator, title=title, markers=True)
    fig.update_layout(xaxis_title="Year", yaxis_title=indicator, hovermode="x unified")
    return fig


def plot_predictions_vs_actuals(
    X_test: pd.DataFrame, y_test: pd.Series, y_pred: pd.Series, title: str, target_column: str
) -> go.Figure:
    """
    Generates a plot comparing actual vs. predicted values.

    Args:
        X_test (pd.DataFrame): The feature data for testing, containing the 'year' column.
        y_test (pd.Series): The actual target values from the test set.
        y_pred (pd.Series): The predicted target values.
        title (str): The title for the plot.
        target_column (str): The name of the target column.

    Returns:
        go.Figure: A Plotly figure object.
    """
    results_df = pd.DataFrame(
        {
            "Year": X_test['year'],
            "Actual": y_test.values,
            "Predicted": y_pred.values,
        }
    )
    results_df = results_df.sort_values(by="Year")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=results_df["Year"],
            y=results_df["Actual"],
            mode="lines+markers",
            name="Actual",
            line=dict(color="blue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=results_df["Year"],
            y=results_df["Predicted"],
            mode="lines+markers",
            name="Predicted",
            line=dict(color="red", dash="dash"),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title=target_column,
        hovermode="x unified",
    )
    return fig