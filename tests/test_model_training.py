# File: tests/test_model_training.py
import pandas as pd
from sklearn.base import is_regressor
from src.model_training import prepare_data, train_model, evaluate_model


def test_prepare_data_split_no_shuffle():
    """
    Tests if data is split correctly for time series (no shuffling).
    """
    # TODO: Create a sample time-series DataFrame.
    # df = pd.DataFrame({'Year': range(2000, 2020), 'Feature': range(20), 'Target': range(20, 40)})
    # Call prepare_data.
    # _, X_test, _, y_test = prepare_data(df, 'Target', ['Year', 'Feature'])
    # Assert that the test set comes after the training set.
    # For example, assert y_test.index.min() > y_train.index.max()
    pass


def test_train_model_returns_fitted_model():
    """
    Tests if train_model returns a fitted regressor model.
    """
    # TODO: Create dummy training data (X_train, y_train).
    # Call train_model with a model name like "Linear Regression".
    # model = train_model(X_train, y_train, "Linear Regression")
    # Assert that the returned object is a regressor.
    # from sklearn.utils.validation import check_is_fitted
    # assert is_regressor(model)
    # check_is_fitted(model) # This will raise an error if the model is not fitted.
    pass


def test_evaluate_model_returns_correct_metrics():
    """
    Tests if the evaluation function returns a dictionary with the correct keys and float values.
    """
    # TODO: Create a mock model and dummy test data (X_test, y_test).
    # class MockModel:
    #     def predict(self, X):
    #         return X.iloc[:, 0] + 1
    # Call evaluate_model.
    # metrics = evaluate_model(MockModel(), X_test, y_test)
    # Assert that the returned dictionary has the keys 'MAE', 'MSE', 'R2 Score'.
    # Assert that all values in the dictionary are floats.
    # assert all(isinstance(v, float) for v in metrics.values())
    pass
