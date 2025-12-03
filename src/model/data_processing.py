import pandas as pd
import numpy as np
from typing import List, Optional
import wbgapi as wb
from abc import ABC, abstractmethod

DEVELOPMENT_MODE = False

# --- Product Interface ---
class DataFetcher(ABC):
    """Abstract Product: Defines the interface for data fetching objects."""
    
    @abstractmethod
    def fetch_data(self, countries: List[str], start_year: int, end_year: int, indicators: dict, 
                   missing_values_threshold: float = 0.3, verbose: bool = False) -> List[pd.DataFrame]:
        pass
    
    @abstractmethod
    def get_source_name(self) -> str:
        pass

    def _handle_missing_values(self, data: pd.DataFrame, threshold: float, verbose: bool = False) -> List[pd.DataFrame]:
        """Shared logic for cleaning data."""
        data_frames = []
        countries = data['country'].unique()
        for country in countries:
            curr_data = data[data['country'] == country]
            curr_missing_percentage = curr_data.isnull().sum() / len(curr_data)
            columns_to_drop = curr_missing_percentage[curr_missing_percentage > threshold].index
            if verbose:
                print(f"Dropping columns for {country}: {list(columns_to_drop)}")
            curr_data = curr_data.drop(columns=columns_to_drop, errors='ignore')
            data_frames.append(curr_data)
        
        for i, curr_data in enumerate(data_frames):
            df = curr_data.copy()
            group_series = df['country']
            others = df.drop(columns=['country'])
            # Forward fill then backward fill
            filled_others = others.groupby(group_series, group_keys=False, sort=False).apply(lambda g: g.ffill().bfill())
            filled = filled_others.copy()
            filled['country'] = group_series
            filled = filled[df.columns]
            data_frames[i] = filled.reset_index(drop=True)
        
        return data_frames


# --- Concrete Product 1: World Bank API ---
class WorldBankAPIFetcher(DataFetcher):
    """Concrete Product for fetching real data."""
    
    def fetch_data(self, countries: List[str], start_year: int, end_year: int, indicators: dict,
                   missing_values_threshold: float = 0.3, verbose: bool = False) -> List[pd.DataFrame]:
        years = list(range(start_year, end_year + 1))
        if len(years) == 0:
            raise ValueError("Input must satisfy: end_year >= start_year.")
        if not countries:
            raise ValueError("Countries list cannot be empty.")

        # Default indicators logic remains here...
        if not indicators:
            indicators = {
                'NY.GDP.MKTP.CD': 'GDP (current US$)',
                'SP.POP.TOTL': 'Population, total',
                'SP.DYN.LE00.IN': 'Life expectancy at birth, total (years)',
                'IT.NET.USER.ZS': 'Individuals using the Internet (% of population)'
            }

        try:
            raw = wb.data.DataFrame(indicators.keys(), countries, time=years, labels=False)
        except Exception as e:
            # Fallback or re-raise 
            raise ConnectionError(f"Failed to connect to World Bank API: {e}")

        raw.index.names = ['country', 'indicator']
        raw = raw.reset_index()

        df = raw.sort_values(['country','indicator'])
        df_melted = pd.melt(df, id_vars=['country', 'indicator'], var_name='year', value_name='value')
        data = df_melted.pivot(index=['country', 'year'], columns='indicator', values='value')
        data = data.reset_index()
        data['year'] = data['year'].str.replace('YR', '').astype(int)
        data = data.rename(columns=indicators)

        return self._handle_missing_values(data, missing_values_threshold, verbose)
    
    def get_source_name(self) -> str:
        return "World Bank API"


# --- Concrete Product 2: Mock Data ---
class MockDataFetcher(DataFetcher):
    """Concrete Product for testing/offline mode."""
    
    def fetch_data(self, countries: List[str], start_year: int, end_year: int, indicators: dict,
                   missing_values_threshold: float = 0.3, verbose: bool = False) -> List[pd.DataFrame]:
        years = list(range(start_year, end_year + 1))
        
        data = []
        for country in countries:
            for year in years:
                row = {'country': country, 'year': year}
                row['GDP (current US$)'] = 1e12 + (year * 1e10)
                row['Population, total'] = 50e6 + (year * 1e5)
                data.append(row)
        
        df = pd.DataFrame(data)
        return self._handle_missing_values(df, missing_values_threshold, verbose)

    def get_source_name(self) -> str:
        return "Mock Data Generator"


# --- Factory Class ---
class DataFetcherFactory:
    """
    Factory Class.
    Responsible for creating the correct DataFetcher instance.
    This replaces the 'Strategy' pattern logic with a Creational pattern.
    """
    
    @staticmethod
    def create_fetcher(dev_mode: bool = False) -> DataFetcher:
        if dev_mode:
            return MockDataFetcher()
        else:
            return WorldBankAPIFetcher()


# --- Facade Function ---
def fetch_world_bank_data(
        countries: List[str], start_year: int, end_year: int, indicators: dict = {},
        missing_values_threshold: float = 0.3, verbose: bool = False
    ) -> List[pd.DataFrame]:
    """
    Client entry point. Uses the Factory to get a fetcher and get data.
    """
    # Use Factory to create the object
    fetcher = DataFetcherFactory.create_fetcher(dev_mode=DEVELOPMENT_MODE)
    
    # Use the object
    return fetcher.fetch_data(countries, start_year, end_year, indicators, missing_values_threshold, verbose)