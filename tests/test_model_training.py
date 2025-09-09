import pytest
import pandas as pd
import numpy as np

# Import validation helpers and model classes from sklearn
from sklearn.base import is_regressor, is_classifier
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

# Import the functions to be tested from your src directory
from src.model_training import (
    select_model,
    prepare_data,
    train_model,
    evaluate_model,
    make_prediction
)

# Define the model names to be used in parameterized tests
# These names must match the keys in the `select_model` function
MODEL_NAMES = [
    "LinearRegression",
    "RandomForestRegressor",
    "GradientBoostingRegressor"
]

@pytest.fixture
def sample_data() -> pd.DataFrame:
    """
    Provides a consistent, small sample DataFrame for use in tests.
    This fixture ensures that all tests run against the same baseline data,
    making them reliable and easy to maintain.
    """
    data = {
        'feature1': np.arange(10),
        'feature2': np.arange(10, 20),
        'target': np.arange(20, 30) * 1.5 + np.random.rand(10),
        'year': np.arange(2010, 2020) # Add a year column for time-series split
    }
    return pd.DataFrame(data)

def test_prepare_data(sample_data):
    """
    Tests the prepare_data function to ensure it splits data correctly.
    Verifies the types, shapes, and ratio of the output splits.
    """
    X_train, X_test, y_train, y_test = prepare_data(sample_data, target_column='target')

    # 1. Check if the function returns four objects
    assert X_train is not None
    assert X_test is not None
    assert y_train is not None
    assert y_test is not None

    # 2. Check the types of the returned objects
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

    # 3. Check if the split ratio is approximately 80/20
    total_rows = len(sample_data)
    assert len(X_train) == total_rows - 5
    assert len(X_test) == 5

    # 4. Check if the number of rows is consistent
    assert len(X_train) + len(X_test) == total_rows
    assert len(y_train) + len(y_test) == total_rows

# FIX: Added the list of model names to the parametrize decorator
@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_select_model(model_name):
    """
    Tests the select_model function for all supported model names.
    Ensures it returns a valid, unfitted scikit-learn regressor object.
    """
    model = select_model(model_name)
    assert model is not None
    # Check if the returned object is a regressor and not a classifier
    assert is_regressor(model)
    assert not is_classifier(model)

def test_select_model_invalid():
    """
    Tests that select_model raises a ValueError for an unsupported model name.
    """
    with pytest.raises(ValueError):
        select_model("InvalidModelName")

# FIX: Added the list of model names to the parametrize decorator
@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_train_and_predict_flow(sample_data, model_name):
    """
    Tests the end-to-end training and prediction flow for each model type.
    This integration test verifies that the functions work together as expected.
    """
    # 1. Prepare data
    X_train, X_test, y_train, _ = prepare_data(sample_data, 'target')
    
    # 2. Select model
    model_instance = select_model(model_name)
    
    # 3. Train model
    trained_model = train_model(model_instance, X_train, y_train)
    
    # FIX: Use scikit-learn's official utility to check if the model is fitted
    try:
        check_is_fitted(trained_model)
    except NotFittedError:
        pytest.fail("Model should be fitted after train_model call.")

    # 4. Make prediction
    prediction = make_prediction(trained_model, X_test)
    
    # Assert that the prediction output has the correct shape
    assert isinstance(prediction, np.ndarray)
    assert len(prediction) == len(X_test)

def test_evaluate_model():
    """
    Tests the evaluate_model function with fixed inputs.
    This test verifies the correctness of the metric calculations by comparing
    the function's output to pre-calculated, known values.
    """
    # Mock a simple model with a predict method
    class MockModel:
        def predict(self, X):
            # Simple prediction: returns the first feature doubled
            return X.iloc[:, 0] * 2

    model = MockModel()
    X_test = pd.DataFrame({
        'feature1': [1.0, 2.0, 3.0, 4.0],
        'feature2': [10.0, 20.0, 30.0, 40.0] 
    })
    y_test = pd.Series([2.5, 3.5, 6.5, 7.5]) # True values

    metrics = evaluate_model(model, X_test, y_test)

    # Pre-calculated expected values
    # y_true = [2.5, 3.5, 6.5, 7.5]
    # y_pred = [2.0, 4.0, 6.0, 8.0]
    # errors = [0.5, -0.5, 0.5, -0.5]
    # abs_errors = [0.5, 0.5, 0.5, 0.5] -> MAE = 0.5
    # sq_errors = [0.25, 0.25, 0.25, 0.25] -> MSE = 0.25
    # R2 = 1 - (SS_res / SS_tot) = 1 - (1.0 / 17.0) = 0.941176...

    assert isinstance(metrics, dict)
    assert 'mae' in metrics
    assert 'mse' in metrics
    assert 'r2_score' in metrics

    assert metrics['mae'] == pytest.approx(0.5)
    assert metrics['mse'] == pytest.approx(0.25)
    # FIX: Corrected the expected R-squared value
    assert metrics['r2_score'] == pytest.approx(0.941176, abs=1e-5)

