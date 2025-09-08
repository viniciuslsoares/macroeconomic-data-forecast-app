import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Literal, List

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Modo de desenvolvimento: usar implementações funcionais simples
DEVELOPMENT_MODE = True

# # A dictionary to map model names to their classes
MODELS = {
    "Linear Regression": LinearRegression,
    "Random Forest": RandomForestRegressor,
    "Gradient Boosting": GradientBoostingRegressor,
}

# Define a type alias for the supported model names for better type checking
ModelChoice = Literal

def select_model(model_name: ModelChoice) -> Any: 
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


def prepare_data(df: pd.DataFrame, target_column: str, features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the data into training and testing sets.

    Args:
        df (pd.DataFrame): The preprocessed DataFrame for a single country.
        target_column (str): The name of the column to be predicted.
        features (List[str]): List of feature column names.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: A tuple containing
        X_train, X_test, y_train, y_test.
    """
    if DEVELOPMENT_MODE:
        # Implementação funcional para desenvolvimento
        df_sorted = df.sort_values(by='year')
        X = df_sorted[features]
        y = df_sorted[target_column]
        return train_test_split(X, y, test_size=0.2, shuffle=False)  # Time series data shouldn't be shuffled
    else:
        # Código original comentado
        pass


def train_model(X_train: pd.DataFrame, y_train: pd.Series, model_name: str) -> Any:
    """
    Trains a specified regression model.

    Args:
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target.
        model_name (str): The name of the model to train (must be a key in MODELS).

    Returns:
        Any: The trained scikit-learn model object.
    """
    if DEVELOPMENT_MODE:
        # Implementação funcional para desenvolvimento
        model = MODELS[model_name]()
        model.fit(X_train, y_train)
        return model
    else:
        # Código original comentado
        pass


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluates the model and returns performance metrics.

    Args:
        model (Any): The trained model object.
        X_test (pd.DataFrame): The testing features.
        y_test (pd.Series): The testing target.

    Returns:
        Dict[str, float]: A dictionary containing MAE, MSE, and R-squared score.
    """
    if DEVELOPMENT_MODE:
        # Implementação funcional para desenvolvimento
        y_pred = model.predict(X_test)
        metrics = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "R2 Score": r2_score(y_test, y_pred)
        }
        return metrics
    else:
        # Código original comentado
        pass


def make_prediction(model: Any, last_known_features: pd.DataFrame) -> float:
    """
    Makes a prediction for the next year based on the last known features.

    Args:
        model (Any): The trained model object.
        last_known_features (pd.DataFrame): A DataFrame with a single row containing
                                            the latest values for the features.

    Returns:
        float: The predicted value for the target variable.
    """
    if DEVELOPMENT_MODE:
        # Implementação funcional para desenvolvimento
        return model.predict(last_known_features)[0]
    else:
        # Código original comentado
        pass