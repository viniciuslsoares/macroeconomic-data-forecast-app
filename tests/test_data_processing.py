import pandas as pd
import numpy as np
from src.model.data_processing import fetch_world_bank_data
import pytest
from unittest.mock import patch


def test_fetch_data_raises_error_on_empty_inputs():
    # empty countries -> ValueError
    with pytest.raises(ValueError, match="Countries list cannot be empty."):
        fetch_world_bank_data(countries=[], start_year=2000, end_year=2010)

    # wrong years -> ValueError
    with pytest.raises(ValueError, match="Input must satisfy: end_year >= start_year."):
        fetch_world_bank_data(
            countries=["BRA"], start_year=2000, end_year=1999)


@patch('wbgapi.data.DataFrame')
def test_get_data_and_handle_missing_values(mock_wb_dataframe):
    """
    End-to-end test for fetch_world_bank_data() and _handle_missing_values().
    Uses mock to avoid real API calls in CI.
    """
    countries = ['BRA', 'NOR']
    years = list(range(1990, 2010))

    # Create mock DataFrame that mimics wbgapi.data.DataFrame structure
    # The structure should have MultiIndex (country, indicator) and columns as years
    indicator_codes = [
        'SP.DYN.LE00.IN', 'SP.DYN.IMRT.IN', 'SH.XPD.CHEX.GD.ZS',
        'NY.GDP.PCAP.CD', 'SI.POV.DDAY', 'SI.POV.2DAY', 'SI.POV.GINI',
        'SE.XPD.TOTL.GD.ZS', 'SE.PRM.CMPT.ZS',
        'SH.STA.SMSS.ZS', 'EG.ELC.ACCS.ZS',
        'EN.ATM.PM25.MC.M3', 'EN.ATM.CO2E.PC', 'AG.LND.FRST.ZS',
        'SP.POP.TOTL', 'SP.URB.GROW',
        'VC.IHR.PSRC.P5'
    ]

    # Create year columns in format 'YR1990', 'YR1991', etc.
    year_columns = [f'YR{year}' for year in years]

    # Build MultiIndex: (country, indicator)
    index_tuples = []
    for country in countries:
        for indicator in indicator_codes:
            index_tuples.append((country, indicator))

    multi_index = pd.MultiIndex.from_tuples(
        index_tuples, names=['country', 'indicator'])

    # Create mock data with realistic values
    mock_data = {}
    for year_col in year_columns:
        year = int(year_col.replace('YR', ''))
        values = []
        for country, indicator in index_tuples:
            if country == 'BRA':
                # Brazil data - simple increasing trends
                if 'SP.POP.TOTL' in indicator:
                    values.append(150000000 + (year - 1990) * 2000000)
                elif 'NY.GDP.PCAP.CD' in indicator:
                    values.append(3000 + (year - 1990) * 200)
                elif 'SP.DYN.LE00.IN' in indicator:
                    values.append(65 + (year - 1990) * 0.3)
                else:
                    values.append(100 + (year - 1990) * 2)
            elif country == 'NOR':
                # Norway data - simple increasing trends
                if 'SP.POP.TOTL' in indicator:
                    values.append(4200000 + (year - 1990) * 30000)
                elif 'NY.GDP.PCAP.CD' in indicator:
                    values.append(25000 + (year - 1990) * 500)
                elif 'SP.DYN.LE00.IN' in indicator:
                    values.append(76 + (year - 1990) * 0.2)
                else:
                    values.append(200 + (year - 1990) * 3)
            else:
                values.append(100)
        mock_data[year_col] = values

    mock_df = pd.DataFrame(mock_data, index=multi_index)
    mock_wb_dataframe.return_value = mock_df

    # Call the function
    results = fetch_world_bank_data(
        countries=countries, start_year=1990, end_year=2010)

    # The implementation builds a sequence and appends per-country DataFrames
    assert len(
        results) == 2, "Should return one cleaned DataFrame per requested country"

    # Find per-country DataFrames
    df_c1 = next(df for df in results if df["country"].iloc[0] == "BRA")
    df_c2 = next(df for df in results if df["country"].iloc[0] == "NOR")

    # After filling, there should be no NaNs in the kept indicator columns
    for df in (df_c1, df_c2):
        # identify indicator columns (exclude 'country' and 'year')
        indicator_cols = [
            c for c in df.columns if c not in ("country", "year")]
        assert len(indicator_cols) > 0
        assert not df[indicator_cols].isnull().values.any(
        ), "Kept indicator columns should be forward/back-filled"
