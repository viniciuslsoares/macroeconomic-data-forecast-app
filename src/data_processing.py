import pandas as pd
import wbgapi as wb
from typing import List


def fetch_world_bank_data(countries: List[str], indicators: dict, start_year: int, end_year: int) -> pd.DataFrame:
    """
    Fetches economic data from the World Bank API for specified countries and indicators.

    Args:
        countries (List[str]): List of country codes (e.g., ['BRA', 'CAN']).
        indicators (dict): Dictionary of indicator codes and their desired names.
        start_year (int): The starting year for the data.
        end_year (int): The ending year for the data.

    Returns:
        pd.DataFrame: A DataFrame with the fetched data, indexed by country and year.
    """
    # TODO: Implement the logic to fetch data using wbgapi and return a pandas DataFrame.
    # df = wb.data.DataFrame(list(indicators.keys()), countries, time=range(start_year, end_year + 1))
    # df_processed = df.stack().unstack(level=1).rename(columns=indicators).reset_index()
    # df_processed['Year'] = pd.to_numeric(df_processed['time'].str.replace('YR', ''))
    # return df_processed.drop(columns=['time'])
    pass


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the input DataFrame by handling missing values.

    Args:
        df (pd.DataFrame): The raw data DataFrame.

    Returns:
        pd.DataFrame: The preprocessed DataFrame with missing values handled using interpolation.
    """
    # TODO: Implement the data cleaning logic. Group by country and then interpolate.
    # df_cleaned = df.groupby('economy').apply(
    #     lambda group: group.interpolate(method='linear', limit_direction='forward', axis=0)
    # ).reset_index(drop=True)
    # return df_cleaned
    pass
