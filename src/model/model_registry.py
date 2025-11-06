from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

"""
Central Model Registry
This dictionary is the single source of truth for all models in the application.
Format:
"Display Name (UI)": (
    "Internal Name (for logs/IDs)",
    ModelClass,
    {default_hyperparameters_dict}
)
"""
MODEL_REGISTRY = {
    "Linear Regression": (
        "LinearRegression",
        LinearRegression,
        {}
    ),
    "Random Forest": (
        "RandomForestRegressor",
        RandomForestRegressor,
        {"random_state": 42}
    ),
    "Gradient Boosting": (
        "GradientBoostingRegressor",
        GradientBoostingRegressor,
        {"random_state": 42}
    )
}


def get_model_names():
    """Returns the list of display names for the UI."""
    return list(MODEL_REGISTRY.keys())


def get_model_instance(display_name: str):
    """Returns an unfitted model instance based on the display name."""
    if display_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{display_name}' not found in registry.")

    _internal_name, model_class, params = MODEL_REGISTRY[display_name]
    return model_class(**params)


def get_internal_model_name(display_name: str):
    """Returns the internal model name based on the display name."""
    if display_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{display_name}' not found in registry.")

    internal_name, _model_class, _params = MODEL_REGISTRY[display_name]
    return internal_name
