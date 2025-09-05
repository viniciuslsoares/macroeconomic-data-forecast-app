import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import pytest
import pandas as pd
from init_data import WBGData

def test_get_data_raises_on_empty_inputs():
    # empty countries -> ValueError
    with pytest.raises(ValueError, match="Countries list cannot be empty"):
        WBGData.get_data(countries=[], years=[2000])

    # empty years -> ValueError
    with pytest.raises(ValueError, match="Years list cannot be empty"):
        WBGData.get_data(countries=["C1"], years=[])


def test_get_data_and_handle_missing_values():
    """
    End-to-end test for get_data() and _handle_missing_values().
    """
    countries = ['BRA', 'NOR']
    years = list(range(1990, 2010))

    results = WBGData.get_data(countries=countries, years=years, missing_values_threshold=0.5, verbose=True)

    # The implementation builds a sequence and appends per-country DataFrames
    assert len(results) == 2, "Should return one cleaned DataFrame per requested country"

    # The class-level attributes should be set
    assert WBGData.countries == countries
    assert WBGData.years == years
    assert isinstance(WBGData.data, pd.DataFrame)

    # Find per-country DataFrames
    df_c1 = next(df for df in results if df["country"].iloc[0] == "BRA")
    df_c2 = next(df for df in results if df["country"].iloc[0] == "NOR")

    # After filling, there should be no NaNs in the kept indicator columns
    for df in (df_c1, df_c2):
        # identify indicator columns (exclude 'country' and 'year')
        indicator_cols = [c for c in df.columns if c not in ("country", "year")]
        assert len(indicator_cols) > 0
        assert not df[indicator_cols].isnull().values.any(), "Kept indicator columns should be forward/back-filled"