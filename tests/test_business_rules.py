import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Project imports
from src.model.model_training import (
    prepare_data, 
    compute_feature_importance, 
    make_recursive_future_prediction
)

# ==============================================================================
# UNIT 1: prepare_data
# CRITERIA: 
#   1. Boundary Value Analysis (BVA): 4 cases (Min, Min+1, Max, Overflow).
#   2. Equivalence Partitioning (EP): 2 classes (Valid Input, Invalid Input).
# RATIONALE: Validates data splitting logic considering the row loss from Lag creation.
# ==============================================================================

class TestPrepareData:
    
    @pytest.fixture
    def time_series_df(self):
        # DataFrame with 10 rows (2010-2019).
        # NOTE: Creating 'lag_1' inside the function will drop the first row (2010).
        # Effective available rows = 9.
        return pd.DataFrame({
            'year': range(2010, 2020),
            'target': range(10),
            'feat': range(10)
        })

    # --- Boundary Value Analysis (BVA) ---

    def test_bva_min_zero(self, time_series_df):
        """[BVA] Min Value: 0 test years. Result: 100% Train."""
        X_train, X_test, _, _ = prepare_data(time_series_df, 'target', test_years_count=0)
        assert len(X_train) == 9 # 10 - 1 (lag)
        assert len(X_test) == 0

    def test_bva_min_plus_one(self, time_series_df):
        """[BVA] Min + 1: 1 test year. Result: Split N-1 Train, 1 Test."""
        X_train, X_test, _, _ = prepare_data(time_series_df, 'target', test_years_count=1)
        assert len(X_train) == 8 # 9 - 1
        assert len(X_test) == 1

    def test_bva_max_exact(self, time_series_df):
        """[BVA] Max Value: test_years == available rows (9). Result: 0 Train, 100% Test."""
        X_train, X_test, _, _ = prepare_data(time_series_df, 'target', test_years_count=9)
        assert len(X_train) == 0
        assert len(X_test) == 9

    def test_bva_overflow(self, time_series_df):
        """[BVA] Overflow: test_years > available rows. Result: 0 Train, 100% Test (Clamped)."""
        X_train, X_test, _, _ = prepare_data(time_series_df, 'target', test_years_count=15)
        assert len(X_train) == 0
        assert len(X_test) == 9

    # --- Equivalence Partitioning (EP) ---

    def test_ep_valid_input(self, time_series_df):
        """[EP] Valid Class: Valid dataframe and target column."""
        X_train, X_test, _, _ = prepare_data(time_series_df, 'target', test_years_count=2)
        # Check structural integrity
        assert 'lag_1' in X_train.columns
        assert 'target' not in X_train.columns

    def test_ep_invalid_input(self, time_series_df):
        """[EP] Invalid Class: Target column does not exist."""
        with pytest.raises(ValueError, match="Target column 'missing_col' not found"):
            prepare_data(time_series_df, 'missing_col', test_years_count=2)


# ==============================================================================
# 2: compute_feature_importance
# CRITERION: Cause-Effect Graph (Decision Table)
# RATIONALE: Validates extraction logic for models inside Pipelines and fallback rules.
# ==============================================================================

class TestFeatureImportanceDecisionTable:
    
    @pytest.fixture
    def sample_X(self):
        return pd.DataFrame({'colA': [10, 20], 'colB': [30, 40]})

    def test_decision_pipeline_extraction(self, sample_X):
        """
        Rule: IF (Model is Pipeline) AND (SHAP works) THEN (Extract final step and transform data).
        """
        # Create a simple real pipeline
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ])
        pipe.fit(sample_X, [1, 2]) # Quick fit

        # Mock SHAP to avoid delay, but validate it received transformed data
        with patch('shap.Explainer') as MockExplainer:
            mock_shap_values = MagicMock()
            # Return simple shap values (dimension 2: samples x features)
            mock_shap_values.values = np.array([[0.5, 0.5], [0.5, 0.5]])
            
            instance = MockExplainer.return_value
            instance.return_value = mock_shap_values
            
            fi = compute_feature_importance(pipe, sample_X)
            
            # Check if SHAP was called with the internal model (LinearRegression), not the Pipeline
            args, _ = MockExplainer.call_args
            assert isinstance(args[0], LinearRegression)
            
            assert not fi.empty

    def test_decision_fallback_to_coef(self, sample_X):
        """
        Rule: IF (SHAP fails) AND (Model has coef_) THEN (Use absolute coef_).
        """
        mock_model = MagicMock()
        mock_model.coef_ = np.array([-0.9, 0.1]) # Coefficients
        del mock_model.feature_importances_ # Ensure it's not a tree

        with patch('shap.Explainer', side_effect=Exception("SHAP died")):
            fi = compute_feature_importance(mock_model, sample_X)
        
        assert fi['colA'] == 0.9 # Absolute value
        assert fi['colB'] == 0.1

    def test_decision_all_fail(self, sample_X):
        """
        Rule: IF (Everything fails) THEN (Return Zeros).
        """
        mock_model = MagicMock(spec=[]) # Empty object

        with patch('shap.Explainer', side_effect=Exception("Error")):
            fi = compute_feature_importance(mock_model, sample_X)
        
        assert fi['colA'] == 0.0
        assert fi['colB'] == 0.0
        
    def test_shap_integration_sanity_check(self, sample_X):
        """
        [Integration Test]: Verifies if the real SHAP can run with the Pipeline
        without crashing and if it returns values in a reasonable scale (not 10^23).
        This ensures the StandardScaler fix is working effectively.
        """
        # 1. Create a Real Pipeline
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ])
        # Train with simple data
        y = [10, 20] 
        pipe.fit(sample_X, y)

        # 2. Call the REAL function (no patch/mock on shap)
        # This exercises the logic: pipe[-1] and pipe[:-1].transform(X)
        fi = compute_feature_importance(pipe, sample_X)

        # 3. Verifications
        assert not fi.empty
        # Verify values are finite
        assert np.all(np.isfinite(fi.values))
        assert fi.max() < 1000.0


# ==============================================================================
# 3: make_recursive_future_prediction
# CRITERION: Decision Table & Equivalence Classes
# RATIONALE: Validate exogenous feature projection logic (Regression vs Static)
# and absolute value reconstruction.
# ==============================================================================

class TestRecursivePredictionLogic:

    @pytest.fixture
    def trained_pipeline(self):
        # Pipeline that predicts a constant difference of +10
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([10.0]) # Predicts Diff = 10
        return mock_model

    def test_reconstruction_accumulated(self, trained_pipeline):
        """
        [Cause-Effect Graph]: Validates mathematical reconstruction.
        Given: Last row (2020) has lag_1 = 100. Model predicts diff = +10.
        
        Logic:
        1. Current Year (2020) Reconstructed = 100 (lag) + 10 (diff) = 110.
           This 110 becomes the 'lag_1' for the next year (2021).
        
        2. Future Year 1 (2021): 110 (lag) + 10 (diff) = 120.
        3. Future Year 2 (2022): 120 (lag) + 10 (diff) = 130.
        """
        last_row = pd.Series({'year': 2020, 'lag_1': 100.0, 'pop': 1000})
        history = pd.DataFrame({'year': [2019, 2020], 'lag_1': [90, 100], 'pop': [1000, 1000]})
        
        future_df = make_recursive_future_prediction(
            trained_pipeline, last_row, history, years_ahead=2
        )
        
        assert len(future_df) == 2
        assert future_df.iloc[0]['year'] == 2021
        assert future_df.iloc[0]['predicted_value'] == 120.0 
        assert future_df.iloc[1]['year'] == 2022
        assert future_df.iloc[1]['predicted_value'] == 130.0 

    def test_feature_projection_logic(self, trained_pipeline):
        """
        [Equivalence Class]: Sufficient History.
        Input: History of 'pop' growing (1000 -> 1100).
        Expected Output: 'pop' in the future should continue growing (Linear Projection), 
        not remain static.
        """
        last_row = pd.Series({'year': 2020, 'lag_1': 100, 'pop': 1100})
        
        # History shows clear growth: 1000 -> 1100 (+100/year)
        history = pd.DataFrame({
            'year': [2019, 2020], 
            'lag_1': [90, 100], 
            'pop': [1000, 1100]
        })

        # Mock the model to ignore prediction, we only want to check features
        trained_pipeline.predict.return_value = np.array([0]) 
        
        # Since the function returns only 'year' and 'predicted_value', 
        # we cannot directly verify the 'pop' feature inside the return dataframe.
        # This test ensures execution without errors when history is valid.
        
        future_df = make_recursive_future_prediction(
            trained_pipeline, last_row, history, years_ahead=1
        )
        
        assert len(future_df) == 1

    def test_feature_projection_insufficient_data(self, trained_pipeline):
        """
        [Equivalence Class]: Insufficient History (Only 1 row).
        Rule: If len(history) < 2, projector fails, must use fallback (static value).
        """
        last_row = pd.Series({'year': 2020, 'lag_1': 100, 'pop': 5000})
        # History with only 1 row -> Impossible to draw a line
        history = pd.DataFrame({'year': [2020], 'lag_1': [100], 'pop': [5000]})
        
        # Should not raise error
        try:
            future_df = make_recursive_future_prediction(
                trained_pipeline, last_row, history, years_ahead=1
            )
        except ValueError:
            pytest.fail("Function crashed with insufficient history (should use fallback)")
            
        assert len(future_df) == 1
        assert future_df.iloc[0]['year'] == 2021