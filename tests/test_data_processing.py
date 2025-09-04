import pandas as pd
import numpy as np
from src.data_processing import preprocess_data


def test_preprocess_data_handles_missing_values():
    """
    Tests if the preprocess_data function correctly fills NaN values using interpolation.
    """
    # TODO: Create a sample DataFrame with NaNs, call preprocess_data, and assert no NaNs remain.
    # data = {'economy': ['BRA', 'BRA'], 'Year': [2000, 2002], 'GDP': [100, 300]}
    # df = pd.DataFrame(data).reindex([0, 1, 2]) # Creates a NaN row
    # df['economy'] = df['economy'].ffill()
    # preprocessed = preprocess_data(df)
    # assert not preprocessed['GDP'].isnull().any()
    pass
