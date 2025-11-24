import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Tuple, Any


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
    fig.update_layout(xaxis_title="Year",
                      yaxis_title=indicator, hovermode="x unified")
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


def prepare_plot_data(model_entry: Dict[str, Any], end_year: int) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepares data for plotting predictions vs actuals with future forecast.

    This function extracts the necessary data from a model entry dictionary,
    generates future predictions, and combines training, test, and future data
    for visualization purposes.

    Args:
        model_entry (Dict[str, Any]): Dictionary containing model artifacts:
            - trained_model: The trained model instance
            - X_train: Training features DataFrame
            - y_train: Training target Series
            - X_test: Test features DataFrame
            - y_test: Test target Series
        end_year (int): The last year in the training data.

    Returns:
        Tuple containing:
        - combined_X (pd.DataFrame): Combined features for train, test, and future years
        - combined_y_actual (pd.Series): Combined actual values (with NaNs for future)
        - combined_y_pred (pd.Series): Combined predicted values for all periods
    """
    # Extract data from model entry
    model = model_entry['trained_model']
    X_train = model_entry['X_train']
    y_train = model_entry['y_train']
    X_test = model_entry['X_test']
    y_test = model_entry['y_test']

    # Generate predictions for training and test sets
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Generate future years for prediction (5-year forecast)
    future_years = pd.DataFrame(
        {'year': range(end_year + 1, end_year + 6)}
    )

    # Populate other features using the last known values from X_test
    other_features = X_test.drop(columns=['year'], errors='ignore').columns
    for feature in other_features:
        future_years[feature] = X_test[feature].iloc[-1]

    # Make predictions for future years
    y_pred_future = model.predict(future_years)

    # Combine X_train, X_test, and future_years for plotting
    combined_X = pd.concat(
        [X_train, X_test, future_years], ignore_index=True
    )

    # Combine y_train (actuals for training), y_test (actuals for test), and NaNs for future years
    combined_y_actual = pd.Series(
        np.concatenate([
            y_train.values,
            y_test.values,
            np.full(len(future_years), np.nan)
        ]),
        index=combined_X.index
    )

    # Combine y_pred_train, y_pred_test, and y_pred_future for plotting
    combined_y_pred = pd.Series(
        np.concatenate([y_pred_train, y_pred_test, y_pred_future]),
        index=combined_X.index
    )

    return combined_X, combined_y_actual, combined_y_pred
