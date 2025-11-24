import pandas as pd
import plotly.graph_objects as go
import io
from datetime import datetime
from src.view.visualization import plot_indicator_trend, plot_predictions_vs_actuals
from src.view.export_utils import (
    export_train_test_data,
    export_actual_vs_predicted_data,
    export_future_predictions,
)


def test_plot_indicator_trend_returns_figure():
    """
    Tests if plot_indicator_trend returns a Plotly Figure object.
    """
    df = pd.DataFrame({"year": [2020, 2021], "GDP": [100, 110]})
    fig = plot_indicator_trend(df, "GDP", "Test Title")
    assert isinstance(fig, go.Figure)


def test_plot_predictions_vs_actuals_returns_figure():
    """
    Tests if plot_predictions_vs_actuals returns a Plotly Figure object.
    """
    X_test = pd.DataFrame({"year": [2020, 2021, 2022]})
    y_test = pd.Series([100, 110, 120])
    y_pred = pd.Series([105, 108, 122])
    fig = plot_predictions_vs_actuals(X_test, y_test, y_pred, "Test Pred Plot", "GDP")
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Test Pred Plot"
    assert fig.layout.yaxis.title.text == "GDP"


def test_export_train_test_data():
    """
    Tests if export_train_test_data returns a BytesIO object with correct CSV content.
    """
    # Create test data
    X_train = pd.DataFrame({"year": [2020, 2021], "feature1": [1, 2]})
    y_train = pd.Series([100, 110])
    X_test = pd.DataFrame({"year": [2022, 2023], "feature1": [3, 4]})
    y_test = pd.Series([120, 130])

    metadata = {
        "model_name": "Test Model",
        "target_column": "GDP",
        "country_name": "Test Country",
        "timestamp": "2023-01-01 12:00:00",
    }

    # Export data
    csv_buffer = export_train_test_data(X_train, y_train, X_test, y_test, metadata)

    # Check if it's a BytesIO object
    assert isinstance(csv_buffer, io.BytesIO)

    # Read the CSV content
    csv_buffer.seek(0)
    csv_content = csv_buffer.read().decode("utf-8")

    # Check if metadata is included
    assert "# Model: Test Model" in csv_content
    assert "# Target Column: GDP" in csv_content
    assert "# Country: Test Country" in csv_content

    # Check if CSV data is present
    assert "year,feature1,target,data_type" in csv_content
    assert "2020,1,100,train" in csv_content
    assert "2022,3,120,test" in csv_content


def test_export_actual_vs_predicted_data():
    """
    Tests if export_actual_vs_predicted_data returns a BytesIO object with correct CSV content.
    """
    # Create test data
    combined_X = pd.DataFrame({"year": [2020, 2021, 2022, 2023]})
    combined_y_actual = pd.Series([100, 110, 120, float("nan")])
    combined_y_pred = pd.Series([105, 108, 122, 135])

    metadata = {
        "model_name": "Test Model",
        "target_column": "GDP",
        "country_name": "Test Country",
        "timestamp": "2023-01-01 12:00:00",
    }

    # Export data
    csv_buffer = export_actual_vs_predicted_data(
        combined_X, combined_y_actual, combined_y_pred, metadata
    )

    # Check if it's a BytesIO object
    assert isinstance(csv_buffer, io.BytesIO)

    # Read the CSV content
    csv_buffer.seek(0)
    csv_content = csv_buffer.read().decode("utf-8")

    # Check if metadata is included
    assert "# Model: Test Model" in csv_content
    assert "# Target Column: GDP" in csv_content
    assert "# Country: Test Country" in csv_content

    # Check if CSV data is present
    assert "year,actual_value,predicted_value,data_type" in csv_content
    assert "2020,100.0,105,actual" in csv_content
    assert "2023,,135,future" in csv_content


def test_export_future_predictions():
    """
    Tests if export_future_predictions returns a BytesIO object with correct CSV content.
    """
    # Create test data
    future_years = pd.DataFrame({"year": [2024, 2025, 2026]})
    y_pred_future = pd.Series([140, 145, 150])

    metadata = {
        "model_name": "Test Model",
        "target_column": "GDP",
        "country_name": "Test Country",
        "timestamp": "2023-01-01 12:00:00",
    }

    # Export data
    csv_buffer = export_future_predictions(future_years, y_pred_future, metadata)

    # Check if it's a BytesIO object
    assert isinstance(csv_buffer, io.BytesIO)

    # Read the CSV content
    csv_buffer.seek(0)
    csv_content = csv_buffer.read().decode("utf-8")

    # Check if metadata is included
    assert "# Model: Test Model" in csv_content
    assert "# Target Column: GDP" in csv_content
    assert "# Country: Test Country" in csv_content

    # Check if CSV data is present
    assert "year,predicted_value" in csv_content
    assert "2024,140" in csv_content
    assert "2025,145" in csv_content
    assert "2026,150" in csv_content
