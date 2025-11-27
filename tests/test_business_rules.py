import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

# Importações do projeto
from src.model.model_training import (
    prepare_data, 
    compute_feature_importance, 
    _prepare_future_features
)

# ==============================================================================
# CRITÉRIO: Análise de Valor Limite (Boundary Value Analysis - BVA)
# JUSTIFICATIVA: Os testes existentes checam o caso médio. 
# Aqui testamos os limites exatos de divisão de dados temporais.
# ==============================================================================

class TestPrepareDataBoundaries:
    
    @pytest.fixture
    def time_series_df(self):
        # DataFrame com 10 anos (2010-2019)
        return pd.DataFrame({
            'year': range(2010, 2020),
            'target': range(10),
            'feat': range(10)
        })

    def test_bva_min_zero_test_years(self, time_series_df):
        """[BVA] Limite Inferior: 0 anos de teste. Todo dataset deve ser treino."""
        X_train, X_test, _, _ = prepare_data(time_series_df, 'target', test_years_count=0)
        
        assert len(X_train) == 10
        assert len(X_test) == 0

    def test_bva_max_all_years_test(self, time_series_df):
        """[BVA] Limite Superior: test_years_count == total de linhas."""
        X_train, X_test, _, _ = prepare_data(time_series_df, 'target', test_years_count=10)
        
        # split_index = 10 - 10 = 0. Tudo vira teste.
        assert len(X_train) == 0
        assert len(X_test) == 10

    def test_bva_overflow_years(self, time_series_df):
        """[BVA] Limite Excedido: test_years_count > total de linhas."""
        # A lógica interna é: split_index = len - count. Se < 0, vira 0.
        X_train, X_test, _, _ = prepare_data(time_series_df, 'target', test_years_count=15)
        
        assert len(X_train) == 0
        assert len(X_test) == 10