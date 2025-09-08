import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Literal, List

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DEVELOPMENT_MODE = True

MODELS = {
    "Linear Regression": LinearRegression,
    "Random Forest": RandomForestRegressor,
    "Gradient Boosting": GradientBoostingRegressor,
}

def select_model(model_name) -> Any: 
    """
    Selects and instantiates a scikit-learn regression model based on its name.

    This function acts as a factory, providing a centralized point for model selection.
    This design makes it easy to add new models in the future without changing
    the training or evaluation logic.

    Args:
        model_name: The name of the model to select. Must be one of the
                    predefined choices in the ModelChoice type.

    Returns:
        An unfitted instance of the selected scikit-learn regressor.

    Raises:
        ValueError: If the model_name is not one of the supported models.
    """
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForestRegressor': RandomForestRegressor(random_state=42),
        'GradientBoostingRegressor': GradientBoostingRegressor(random_state=42)
    }
    
    model = models.get(model_name)
    if model is None:
        raise ValueError(f"Invalid model name: {model_name}. "
                         f"Choose from {list(models.keys())}")
    return model


def prepare_data(
    df: pd.DataFrame, 
    target_column: str, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple:
    """
    Prepares the dataset for model training by separating features and target,
    and splitting them into training and testing sets.

    Args:
        df: The preprocessed time-series dataset.
        target_column: The name of the column to be used as the target variable (y).
        test_size: The proportion of the dataset to allocate to the test set.
        random_state: A seed for the random number generator to ensure reproducibility.

    Returns:
        A tuple containing four elements:
        - X_train: Features for the training set.
        - X_test: Features for the testing set.
        - y_train: Target variable for the training set.
        - y_test: Target variable for the testing set.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

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