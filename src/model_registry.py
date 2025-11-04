# src/model_registry.py
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

"""
Central Model Registry
Este dicionário é a fonte única da verdade para todos os modelos no aplicativo.
Formato:
"Nome de Exibição (UI)": (
    "Nome Interno (para logs/IDs)",
    ClasseDoModelo,
    {dicionário_de_hiperparâmetros_padrão}
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
    """Retorna a lista de nomes de exibição para a UI."""
    return list(MODEL_REGISTRY.keys())


def get_model_instance(display_name: str):
    """Retorna uma instância não treinada do modelo com base no nome de exibição."""
    if display_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Modelo '{display_name}' não encontrado no registro.")

    _internal_name, model_class, params = MODEL_REGISTRY[display_name]
    return model_class(**params)


def get_internal_model_name(display_name: str):
    """Retorna o nome interno do modelo com base no nome de exibição."""
    if display_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Modelo '{display_name}' não encontrado no registro.")

    internal_name, _model_class, _params = MODEL_REGISTRY[display_name]
    return internal_name
