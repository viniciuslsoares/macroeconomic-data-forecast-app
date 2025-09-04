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
    # TODO: Create a line plot using Plotly Express.
    # fig = px.line(df, x='Year', y=indicator, title=title, markers=True)
    # return fig
    pass


def plot_predictions_vs_actuals(y_test: pd.Series, y_pred: pd.Series, title: str) -> go.Figure:
    """
    Generates a plot comparing actual vs. predicted values.

    Args:
        y_test (pd.Series): The actual target values from the test set.
        y_pred (pd.Series): The predicted target values.
        title (str): The title for the plot.

    Returns:
        go.Figure: A Plotly figure object.
    """
    # TODO: Create a line plot comparing actuals and predictions.
    # results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    # fig = px.line(results_df, y=['Actual', 'Predicted'], title=title)
    # fig.update_traces(marker=dict(size=8))
    # return fig
    pass
