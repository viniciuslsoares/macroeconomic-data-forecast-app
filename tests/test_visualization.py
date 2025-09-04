# File: tests/test_visualization.py
import pandas as pd
import plotly.graph_objects as go
from src.visualization import plot_indicator_trend, plot_predictions_vs_actuals


def test_plot_indicator_trend_returns_figure():
    """
    Tests if plot_indicator_trend returns a Plotly Figure object.
    """
    # TODO: Create a sample DataFrame.
    # df = pd.DataFrame({'Year': [2020, 2021], 'GDP': [100, 110]})
    # Call the function.
    # fig = plot_indicator_trend(df, 'GDP', 'Test Title')
    # Assert that the returned object is an instance of go.Figure.
    # assert isinstance(fig, go.Figure)
    pass


def test_plot_predictions_vs_actuals_returns_figure():
    """
    Tests if plot_predictions_vs_actuals returns a Plotly Figure object.
    """
    # TODO: Create sample Series for y_test and y_pred.
    # y_test = pd.Series([100, 110, 120])
    # y_pred = pd.Series([105, 108, 122])
    # Call the function.
    # fig = plot_predictions_vs_actuals(y_test, y_pred, 'Test Pred Plot')
    # Assert that the returned object is an instance of go.Figure.
    # assert isinstance(fig, go.Figure)
    pass

