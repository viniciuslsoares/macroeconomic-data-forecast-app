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
        
        
# ==============================================================================
# UNIDADE 2: compute_feature_importance
# CRITÉRIO: Tabela de Decisão (Grafo Causa-Efeito)
# JUSTIFICATIVA: A função possui lógica de fallback complexa (Try SHAP -> Try Tree -> Try Linear -> Zeros).
# ==============================================================================

class TestFeatureImportanceDecisionTable:
    
    @pytest.fixture
    def sample_X(self):
        return pd.DataFrame({'colA': [1, 2], 'colB': [3, 4]})

    def test_decision_rule_shap_fails_tree_succeeds(self, sample_X):
        """
        Regra: SE (SHAP falha) E (Modelo tem feature_importances_) ENTÃO (Usa feature_importances_).
        """
        mock_model = MagicMock()
        # Simula atributo de Árvore
        mock_model.feature_importances_ = np.array([0.8, 0.2])
        # Garante que não tem coef
        del mock_model.coef_ 

        # Força erro no SHAP (simulando falha na biblioteca ou cálculo)
        with patch('shap.Explainer', side_effect=Exception("SHAP Error")):
            fi = compute_feature_importance(mock_model, sample_X)
        
        assert fi['colA'] == 0.8
        assert fi['colB'] == 0.2

    def test_decision_rule_shap_fails_linear_succeeds(self, sample_X):
        """
        Regra: SE (SHAP falha) E (Não tem feature_importances_) E (Tem coef_) ENTÃO (Usa coef_ absoluto).
        """
        mock_model = MagicMock()
        # Simula Regressão Linear (coeficientes negativos devem virar absolutos)
        mock_model.coef_ = np.array([-0.5, 0.5])
        # Garante que não tem feature_importances_
        del mock_model.feature_importances_

        with patch('shap.Explainer', side_effect=Exception("SHAP Error")):
            fi = compute_feature_importance(mock_model, sample_X)
        
        assert fi['colA'] == 0.5 # abs(-0.5)
        assert fi['colB'] == 0.5

    def test_decision_rule_all_fail(self, sample_X):
        """
        Regra: SE (Tudo falha ou atributos inexistentes) ENTÃO (Retorna Zeros).
        """
        mock_model = MagicMock(spec=[]) # Modelo vazio sem atributos

        with patch('shap.Explainer', side_effect=Exception("SHAP Error")):
            fi = compute_feature_importance(mock_model, sample_X)
        
        assert fi['colA'] == 0.0
        assert fi['colB'] == 0.0
        

# ==============================================================================
# UNIDADE 3: _prepare_future_features
# CRITÉRIO: Particionamento em Classes de Equivalência (EP)
# JUSTIFICATIVA: Validar como o sistema preenche dados futuros baseados na existência
# ou ausência da feature no dataset de origem.
# ==============================================================================

class TestPrepareFutureFeaturesEP:

    def test_ep_valid_feature_propagation(self):
        """
        [Classe Válida]: A feature existe no DataFrame fonte.
        Resultado esperado: O valor do último ano é propagado para o ano seguinte.
        """
        source_df = pd.DataFrame({
            'year': [2018],
            'gdp': [1000],
            'pop': [500]
        })
        
        future_df = _prepare_future_features(source_df, end_year=2018)
        
        # Verifica se criou o ano seguinte
        assert future_df.iloc[0]['year'] == 2019
        # Verifica se propagou os valores da última linha
        assert future_df.iloc[0]['gdp'] == 1000 
        assert future_df.iloc[0]['pop'] == 500

    def test_ep_robustness_on_subset(self):
        """
        [Classe de Robustez]: Garante que a função opera corretamente mesmo com 
        um subconjunto limitado de colunas, verificando se o loop interno 
        identifica apenas as colunas presentes.
        """
        # Cria um DF com apenas uma coluna de feature além do ano
        source_df = pd.DataFrame({'year': [2018], 'gdp': [100]})
        
        future_df = _prepare_future_features(source_df, end_year=2018)
        
        # Verifica se a saída contém apenas as colunas esperadas e propagou o valor
        assert 'gdp' in future_df.columns
        assert 'pop' not in future_df.columns  # Garante que não inventou colunas
        assert future_df.iloc[0]['gdp'] == 100
        
