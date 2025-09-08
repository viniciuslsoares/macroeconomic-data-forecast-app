import pandas as pd
import plotly.graph_objects as go
from src.visualization import plot_indicator_trend, plot_predictions_vs_actuals

def test_plot_indicator_trend_returns_figure():
    """
    Tests if plot_indicator_trend returns a Plotly Figure object.
    """
    df = pd.DataFrame({'year': [2020, 2021], 'GDP': [100, 110]})
    fig = plot_indicator_trend(df, 'GDP', 'Test Title')
    assert isinstance(fig, go.Figure)

def test_plot_predictions_vs_actuals_returns_figure():
    """
    Tests if plot_predictions_vs_actuals returns a Plotly Figure object.
    """
    y_test = pd.Series([100, 110, 120])
    y_pred = pd.Series([105, 108, 122])
    fig = plot_predictions_vs_actuals(y_test, y_pred, 'Test Pred Plot', 'GDP')
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == 'Test Pred Plot'
    assert fig.layout.yaxis.title.text == 'GDP'