import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
from sklearn.linear_model import LinearRegression 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline 
import shap
import sys
import io
import joblib
sys.path.append(".")
from src.model.model_registry import get_model_instance

def compute_feature_importance(model: Any, X: pd.DataFrame) -> pd.Series:
    """
    Computes and ranks feature importance scores using SHAP or model attributes.

    This function prioritizes model-agnostic interpretation via SHAP values to provide
    consistent insights across different model types. It employs a robust fallback
    mechanism: if SHAP calculation fails, it reverts to native model attributes
    (such as feature_importances_ or coef_) to ensure a result is always returned.

    Args:
        model: The trained machine learning model to inspect.
        X: The input DataFrame used for calculating importance. For efficiency,
           this data is sampled before generating SHAP values.

    Returns:
        A pandas Series mapping feature names to their absolute importance scores,
        sorted in descending order. Returns a Series of zeros if importance
        cannot be determined.
    """
    if X is None or X.shape[0] == 0:
        return pd.Series(dtype=float)

    if isinstance(model, Pipeline):
        try:
            preprocessor = model[:-1]
            estimator = model[-1]
            
            X_transformed_array = preprocessor.transform(X)
            X_for_shap = pd.DataFrame(X_transformed_array, columns=X.columns)
            model_to_explain = estimator
        except Exception as e:
            print(f"Pipeline parsing failed: {e}")
            model_to_explain = model
            X_for_shap = X
    else:
        model_to_explain = model
        X_for_shap = X

    feature_names = list(X.columns)

    # try SHAP
    try:
        # sample for speed/stability
        sample = X_for_shap if len(X_for_shap) <= 200 else X_for_shap.sample(200, random_state=42)
        # Use shap.Explainer
        explainer = shap.Explainer(model_to_explain, sample)
        shap_values = explainer(sample)
        
        # shap can return list for multi-output; try to combine
        vals = shap_values.values
        if isinstance(vals, list): vals = np.array(vals)
        
        if vals.ndim == 3:
            vals = np.mean(np.abs(vals), axis=1) 
        elif vals.ndim == 2:
            vals = np.abs(vals)
        
        mean_abs = np.mean(vals, axis=0)
        
        fi = pd.Series(mean_abs, index=feature_names)
        return fi.sort_values(ascending=False)

    except Exception as e:
        print(f"SHAP failed: {e}. Falling back to native importance.")
        try:
            if hasattr(model_to_explain, "feature_importances_"):
                return pd.Series(np.abs(model_to_explain.feature_importances_), index=feature_names).sort_values(ascending=False)
            elif hasattr(model_to_explain, "coef_"):
                return pd.Series(np.abs(model_to_explain.coef_), index=feature_names).sort_values(ascending=False)
        except:
            pass
            
        return pd.Series(0, index=feature_names)

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
    Prepares the dataset for training by applying stationarity transformations.

    Instead of predicting absolute values directly, this function transforms the 
    problem into predicting the year-over-year difference (differentiation). 
    It creates a 'lag_1' feature (previous year's value) and a 'target_diff' 
    target variable.

    Transformation Logic:
        y_t (target) = Value_t - Value_{t-1}
        Feature 'lag_1' = Value_{t-1}

    Args:
        df: The preprocessed time-series dataset.
        target_column: The name of the column to be used as the target variable.
        test_years_count: The number of most recent years to allocate to the test set.

    Returns:
        A tuple containing four elements:
        - X_train: Features (including 'lag_1') for the training set.
        - X_test: Features (including 'lag_1') for the testing set.
        - y_train: Target variable (Difference) for the training set.
        - y_test: Target variable (Difference) for the testing set.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    df_sorted = df.sort_values(by='year').copy()

    # Feature Engineering
    df_sorted['lag_1'] = df_sorted[target_column].shift(1)
    
    df_sorted['target_diff'] = df_sorted[target_column].diff()

    df_sorted = df_sorted.dropna()

    # Split
    split_index = len(df_sorted) - test_years_count
    if split_index < 0: split_index = 0

    X = df_sorted.drop(columns=[target_column, 'target_diff'])
    y = df_sorted['target_diff'] 

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test


def train_model(model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    Trains a machine learning model wrapped in a preprocessing pipeline.

    This function automatically wraps the provided model with a StandardScaler
    to ensure robust performance for scale-sensitive algorithms (like SVR or 
    Linear Regression) and to handle the magnitude differences between 
    'lag_1' (e.g., Trillions) and other features (e.g., percentages).

    Args:
        model: An unfitted instance of a scikit-learn model.
        X_train: The feature data for training.
        y_train: The target data (differences) for training.

    Returns:
        A fitted sklearn.pipeline.Pipeline containing [StandardScaler, Model].
    """
    pipeline = make_pipeline(StandardScaler(), model)
    pipeline.fit(X_train, y_train)
    return pipeline


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


def save_model(model: Any, filepath: str = None) -> bytes:
    """
    Save a trained model to disk or return as bytes for download.
    
    Args:
        model: A trained scikit-learn model.
        filepath: Optional file path to save the model. If None, returns bytes.
        
    Returns:
        Bytes representation of the model if filepath is None.
    """
    if filepath:
        joblib.dump(model, filepath)
        return None
    else:
        # Save to bytes for download
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        return buffer.getvalue()


def load_model(filepath: str) -> Any:
    """
    Load a trained model from disk.
    
    Args:
        filepath: Path to the saved model file.
        
    Returns:
        The loaded trained model.
    """
    return joblib.load(filepath)


def evaluate_loaded_model(
    model: Any, 
    country_data: pd.DataFrame, 
    target_column: str, 
    end_year: int
) -> dict:
    """
    Evaluates a loaded model by reconstructing absolute values from its difference predictions.

    This function ensures that a model loaded from a file (which predicts differences)
    is correctly evaluated against absolute historical data. It performs the same
    reconstruction logic ($Abs = Lag + Diff$) as the training pipeline to generate
    valid metrics and plotting data.

    Args:
        model: A loaded trained model (or Pipeline).
        country_data: DataFrame containing the country data.
        target_column: Name of the target column.
        end_year: Last year in the training data.

    Returns:
        Dictionary containing artifacts compatible with the dashboard (metrics, 
        reconstructed predictions, future forecast).
    """
    # 1. Prepare data (remove 'country' column if it exists)
    if 'country' in country_data.columns:
        country_data_for_training = country_data.drop(columns=['country'])
    else:
        country_data_for_training = country_data

    X_train, X_test, y_train_diff, y_test_diff = prepare_data(
        country_data_for_training, target_column
    )

    # 2. Evaluate Model    
    y_pred_train_diff = model.predict(X_train)
    y_pred_train_abs = X_train['lag_1'] + y_pred_train_diff
    y_train_abs = X_train['lag_1'] + y_train_diff

    y_pred_test_diff = model.predict(X_test)
    y_pred_test_abs = X_test['lag_1'] + y_pred_test_diff
    y_test_abs = X_test['lag_1'] + y_test_diff
    
    # 3. Prepare Future Features
    metrics = {
        'mae': mean_absolute_error(y_test_abs, y_pred_test_abs),
        'mse': mean_squared_error(y_test_abs, y_pred_test_abs),
        'r2_score': r2_score(y_test_abs, y_pred_test_abs)
    }

    # 4. Make Prediction
    last_features = X_test.iloc[-1]
    X_history = pd.concat([X_train, X_test])
    
    model_step = model.steps[1][1] if hasattr(model, 'steps') else model

    future_df = make_recursive_future_prediction(model, last_features, X_history, years_ahead=5)

    # 5. Return all artifacts
    return {
        "trained_model": model,
        "metrics": metrics,
        "prediction": future_df.iloc[0]['predicted_value'],
        "X_train": X_train,
        "y_train": y_train_abs,   
        "X_test": X_test,
        "y_test": y_test_abs,     
        "y_pred_train_abs": y_pred_train_abs,
        "y_pred_test_abs": y_pred_test_abs,
        "target_column": target_column
    }


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

    # Reorder columns to match the training data order
    next_year_features = next_year_features[features_source_df.columns]
    
    return next_year_features


def run_training_pipeline(
    country_data: pd.DataFrame, 
    target_column: str, 
    model_name: str, 
    end_year: int
) -> dict:
    r"""
    Executes the training pipeline using the Difference/Lag strategy and reconstructs absolute values.

    Workflow:
    1. Prepares data by calculating differences and lags.
    2. Trains the model to predict the difference.
    3. RECONSTRUCTION: Calculates absolute predictions for Train/Test sets 
       by summing the predicted difference with the actual Lag value:
       $\hat{y}_{abs} = Lag + \hat{y}_{diff}$
    4. Generates recursive future predictions.
    5. Computes feature importance using the full pipeline.

    Args:
        country_data: Raw DataFrame for a specific country.
        target_column: The name of the target variable.
        model_name: The name of the model to instantiate.
        end_year: The split year (metadata).

    Returns:
        Dictionary containing artifacts for the frontend, including:
        - 'metrics': Calculated on ABSOLUTE values (MAE, MSE, R2).
        - 'y_pred_train_abs': Reconstructed absolute predictions for training set.
        - 'y_pred_test_abs': Reconstructed absolute predictions for test set.
        - 'prediction': Future prediction for the immediate next year.
    """

    # 1. Prepare data (remove 'country' column if it exists)
    # Target is now DIFF
    if 'country' in country_data.columns:
        country_data = country_data.drop(columns=['country'])

    # Creates 'target_diff' e 'lag_1'
    X_train, X_test, y_train_diff, y_test_diff = prepare_data(country_data, target_column)

    # 2. Select and Train Model
    unfitted_model = select_model(model_name)
    model = train_model(unfitted_model, X_train, y_train_diff)

    # 3. RECONSTRUCTION OF ABSOLUTE VALUES (Chart Correction)
    # Predicted Absolute Value = Lag (Previous Actual Value) + Predicted Difference
    
    y_pred_train_diff = model.predict(X_train)
    y_pred_train_abs = X_train['lag_1'] + y_pred_train_diff 
    y_train_abs = X_train['lag_1'] + y_train_diff #

    y_pred_test_diff = model.predict(X_test)
    y_pred_test_abs = X_test['lag_1'] + y_pred_test_diff
    y_test_abs = X_test['lag_1'] + y_test_diff

    metrics = {
        'mae': mean_absolute_error(y_test_abs, y_pred_test_abs),
        'mse': mean_squared_error(y_test_abs, y_pred_test_abs),
        'r2_score': r2_score(y_test_abs, y_pred_test_abs)
    }

    # 4. Predição Futura Recursiva (Já estava correta, mas mantemos aqui)
    last_features = X_test.iloc[-1]
    X_history = pd.concat([X_train, X_test])
    
    future_df = make_recursive_future_prediction(model, last_features, X_history, years_ahead=5)
    
    # 5. Feature Importance
    try:
        feature_importance = compute_feature_importance(model, X_train)
    except Exception as e:
        print(f"Feature importance error: {e}")
        feature_importance = pd.Series(dtype=float)

    return {
        "trained_model": model,
        "metrics": metrics,
        "prediction": future_df.iloc[0]['predicted_value'],
        "X_train": X_train,
        "y_train": y_train_abs,  
        "X_test": X_test,
        "y_test": y_test_abs,     
        "y_pred_train_abs": y_pred_train_abs,
        "y_pred_test_abs": y_pred_test_abs,
        "target_column": target_column,
        "feature_importance": feature_importance
    }


def make_recursive_future_prediction(
    model: Any, 
    last_known_row: pd.Series, 
    X_history: pd.DataFrame, 
    years_ahead: int = 5
) -> pd.DataFrame:
    r"""
    Generates future predictions using a recursive autoregressive strategy.

    Since the model predicts the *difference* ($y_t - y_{t-1}$), this function:
    1. Predicts the difference for year $t+1$.
    2. Reconstructs the absolute value: $Abs_{t+1} = Abs_t + PredDiff$.
    3. Updates the 'lag_1' feature for the next iteration using the calculated $Abs_{t+1}$.
    4. Projects exogenous features (e.g., Population) using linear trends derived 
       from X_history to avoid static 'flatline' inputs.

    Args:
        model: The trained pipeline (predicting differences).
        last_known_row: The last row of data from the test set (context).
        X_history: Historical feature data used to train trend projectors for exogenous variables.
        years_ahead: Number of years to forecast (default: 5).

    Returns:
        A DataFrame containing 'year' and 'predicted_value' (Absolute) for the forecast horizon.
    """
    future_predictions = []
    
    current_features = last_known_row.copy()
    
    pred_diff_initial = model.predict(pd.DataFrame([current_features]))[0]
    current_absolute_value = current_features['lag_1'] + pred_diff_initial
    
    current_year = int(current_features['year'])

    feature_projectors = {}
    exogenous_features = [c for c in X_history.columns if c not in ['year', 'lag_1']]
    
    for feat in exogenous_features:
        # Use only needed columns and avoid warning
        hist_data = X_history[['year', feat]].tail(10).dropna()
        
        if len(hist_data) > 1: 
            lr = LinearRegression()
            lr.fit(hist_data[['year']], hist_data[feat])
            feature_projectors[feat] = lr
        else:
            feature_projectors[feat] = None

    for i in range(years_ahead):
        next_year = current_year + 1
        next_row = current_features.copy()
        next_row['year'] = next_year
        
        next_row['lag_1'] = current_absolute_value
        
        for feat in exogenous_features:
            projector = feature_projectors.get(feat)
            if projector:
                # FIX: Pass DataFrame with feature name to avoid warning
                next_val = projector.predict(pd.DataFrame({'year': [next_year]}))[0]
                next_row[feat] = next_val
            else:
                next_row[feat] = current_features[feat]

        pred_diff = model.predict(pd.DataFrame([next_row]))[0]
        
        next_absolute_value = current_absolute_value + pred_diff
        
        future_predictions.append({
            'year': next_year,
            'predicted_value': next_absolute_value
        })
        
        current_absolute_value = next_absolute_value
        current_year = next_year
        current_features = next_row

    return pd.DataFrame(future_predictions)