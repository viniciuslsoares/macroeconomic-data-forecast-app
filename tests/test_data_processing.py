import pandas as pd
import numpy as np
from src.model.data_processing import fetch_world_bank_data
import pytest

def test_fetch_data_raises_error_on_empty_inputs():
    # empty countries -> ValueError
    with pytest.raises(ValueError, match="Countries list cannot be empty."):
        fetch_world_bank_data(countries=[], start_year=2000, end_year=2010)

    # wrong years -> ValueError
    with pytest.raises(ValueError, match="Input must satisfy: end_year >= start_year."):
        fetch_world_bank_data(countries=["BRA"], start_year=2000, end_year=1999)

def test_get_data_and_handle_missing_values():
    """
    End-to-end test for fetch_world_bank_data() and _handle_missing_values().
    """
    countries = ['BRA', 'NOR']
    years = list(range(1990, 2010))

    results = fetch_world_bank_data(countries=countries, start_year=1990, end_year=2010)

    # The implementation builds a sequence and appends per-country DataFrames
    assert len(results) == 2, "Should return one cleaned DataFrame per requested country"

    # Find per-country DataFrames
    df_c1 = next(df for df in results if df["country"].iloc[0] == "BRA")
    df_c2 = next(df for df in results if df["country"].iloc[0] == "NOR")

    # After filling, there should be no NaNs in the kept indicator columns
    for df in (df_c1, df_c2):
        # identify indicator columns (exclude 'country' and 'year')
        indicator_cols = [c for c in df.columns if c not in ("country", "year")]
        assert len(indicator_cols) > 0
        assert not df[indicator_cols].isnull().values.any(), "Kept indicator columns should be forward/back-filled"