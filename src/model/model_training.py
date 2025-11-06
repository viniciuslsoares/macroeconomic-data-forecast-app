import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Literal, List

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
sys.path.append(".")
from src.model.model_registry import get_model_instance


def select_model(model_name) -> Any:
    """
    Selects and instantiates a scikit-learn regression model based on its name.

    This function acts as a factory, providing a centralized point for model selection.
    This design makes it easy to add new models in the future without changing
    the training or evaluation logic.

    Args:
        model_name: The display name of the model to select (e.g., "Linear Regression").

    Returns:
        An unfitted instance of the selected scikit-learn regressor.

    Raises:
        ValueError: If the model_name is not one of the supported models.
    """
    return get_model_instance(model_name)


def prepare_data(
    df: pd.DataFrame,
    target_column: str,
    test_years_count: int = 5,
) -> Tuple:
    """
    Prepares the dataset for model training by separating features and target,
    and splitting them into training and testing sets using a time-series split.

    Args:
        df: The preprocessed time-series dataset.
        target_column: The name of the column to be used as the target variable (y).
        test_years_count: The number of most recent years to allocate to the test set.

    Returns:
        A tuple containing four elements:
        - X_train: Features for the training set.
        - X_test: Features for the testing set.
        - y_train: Target variable for the training set.
        - y_test: Target variable for the testing set.
    """
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in DataFrame.")

    df_sorted = df.sort_values(by='year')

    # Determine the split point
    split_index = len(df_sorted) - test_years_count
    if split_index < 0:  # Handle cases where not enough data for test_years_count
        split_index = 0  # Use all data for training, no test set

    X = df_sorted.drop(columns=[target_column])
    y = df_sorted[target_column]

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test


def train_model(model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """
    Trains (fits) a machine learning model on the training data.

    Args:
        model: An unfitted instance of a scikit-learn model.
        X_train: The feature data for training.
        y_train: The target data for training.

    Returns:
        The trained (fitted) model object.
    """
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluates the performance of a trained model on the unseen test set.

    Calculates Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R^2).

    Args:
        model: A trained scikit-learn model.
        X_test: The feature data for testing.
        y_test: The true target data for testing.

    Returns:
        A dictionary containing the calculated performance metrics.
    """
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        'mae': mae,
        'mse': mse,
        'r2_score': r2
    }
    return metrics


def make_prediction(model: Any, X_new: pd.DataFrame) -> np.ndarray:
    """
    Generates a prediction for new, unseen feature data.

    Args:
        model: A trained scikit-learn model.
        X_new: A DataFrame containing the feature values for which to predict.
               This should have the same columns as the training data.

    Returns:
        A numpy array containing the predictions.
    """
    return model.predict(X_new)


def _prepare_future_features(features_source_df: pd.DataFrame, end_year: int) -> pd.DataFrame:
    """
    Prepares a DataFrame with features for next year's prediction.

    This is a private helper function that engineers the necessary features
    to make future predictions, using the last known values.

    Args:
        features_source_df: DataFrame containing the most recent features (usually X_test or X_train).
        end_year: The last year in the training data.

    Returns:
        DataFrame with a single row containing features for the following year.
    """
    # Prepare features for the next year (end_year + 1)
    next_year_features = pd.DataFrame({'year': [end_year + 1]})

    # Populate other features using the last known values from the source
    other_features = features_source_df.drop(
        columns=['year'], errors='ignore').columns
    for feature in other_features:
        if feature in features_source_df.columns:
            next_year_features[feature] = features_source_df[feature].iloc[-1]
        else:
            # Fallback if the column does not exist
            next_year_features[feature] = 0

    return next_year_features


def run_training_pipeline(country_data: pd.DataFrame, target_column: str, model_name: str, end_year: int) -> dict:
    """
    Executes the complete training, evaluation, and prediction pipeline.

    This function orchestrates the entire ML workflow, from data preparation
    to final prediction, encapsulating all ML business logic.

    Args:
        country_data: DataFrame containing the country data to be analyzed.
        target_column: Name of the column that will be the prediction target.
        model_name: Display name of the model to be used (e.g., "Linear Regression").
        end_year: Last year present in the training data.

    Returns:
        Dictionary containing all artifacts generated by the pipeline:
        - trained_model: The trained model
        - metrics: Dictionary with evaluation metrics (MAE, MSE, R²)
        - prediction: Predicted value for the next year
        - X_train, y_train: Training data
        - X_test, y_test: Test data
        - target_column: Target column name (for reference)
    """

    # 1. Prepare data (remove 'country' column if it exists)
    if 'country' in country_data.columns:
        country_data_for_training = country_data.drop(columns=['country'])
    else:
        country_data_for_training = country_data

    X_train, X_test, y_train, y_test = prepare_data(
        country_data_for_training, target_column
    )

    # 2. Select and Train Model
    unfitted_model = select_model(model_name)
    model = train_model(unfitted_model, X_train, y_train)

    # 3. Evaluate Model
    metrics = evaluate_model(model, X_test, y_test)

    # 4. Prepare Future Features
    # Determine the source: X_test if not empty, otherwise X_train
    features_source_df = X_test if not X_test.empty else X_train
    next_year_features = _prepare_future_features(features_source_df, end_year)

    # 5. Make Prediction
    prediction = make_prediction(model, next_year_features)

    # 6. Return all artifacts
    return {
        "trained_model": model,
        "metrics": metrics,
        "prediction": prediction[0],
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "target_column": target_column
    }
