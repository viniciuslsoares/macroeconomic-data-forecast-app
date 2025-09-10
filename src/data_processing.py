import pandas as pd
import numpy as np
from typing import List
import wbgapi as wb


DEVELOPMENT_MODE = False

def fetch_world_bank_data(
        countries: List[str], start_year: int, end_year: int, indicators: dict = {},
        missing_values_threshold: int = 0.3, verbose: bool = False
    ) -> List[pd.DataFrame]:
    """
    Fetches economic data from the World Bank API for specified countries and indicators.
    
    Args:
        countries (List[str]): List of country codes (e.g., ['BRA', 'CAN']).
        indicators (dict): Dictionary of indicator codes and their desired names.
        start_year (int): The starting year for the data.
        end_year (int): The ending year for the data.

    Returns:
        list[pd.DataFrame]: A list of DataFrames. Each represents the processed fetched data for each given country.
    """
    years = list(range(start_year, end_year + 1))
    if len(years) == 0:
        raise ValueError("Input must satisfy: end_year >= start_year.")
    if DEVELOPMENT_MODE:
        
        
        
        data = []
        
        for country in countries:
            for year in years:
                row = {'economy': country, 'Year': year}
                
                if country == 'BRA':
                    row['GDP (current US$)'] = 1.8e12 + (year - 2000) * 5e10 + np.random.normal(0, 1e11)
                    row['Population, total'] = 200e6 + (year - 2000) * 1.5e6 + np.random.normal(0, 1e6)
                    row['CO2 emissions (kt)'] = 400000 + (year - 2000) * 5000 + np.random.normal(0, 20000)
                    row['Life expectancy at birth, total (years)'] = 70 + (year - 2000) * 0.2 + np.random.normal(0, 0.5)
                    row['Individuals using the Internet (% of population)'] = min(95, 10 + (year - 2000) * 3.5 + np.random.normal(0, 2))
                
                elif country == 'CAN':
                    row['GDP (current US$)'] = 1.2e12 + (year - 2000) * 3e10 + np.random.normal(0, 5e10)
                    row['Population, total'] = 31e6 + (year - 2000) * 0.8e6 + np.random.normal(0, 0.5e6)
                    row['CO2 emissions (kt)'] = 550000 + (year - 2000) * -2000 + np.random.normal(0, 15000)
                    row['Life expectancy at birth, total (years)'] = 79 + (year - 2000) * 0.1 + np.random.normal(0, 0.3)
                    row['Individuals using the Internet (% of population)'] = min(98, 50 + (year - 2000) * 2 + np.random.normal(0, 1))
                
                
                if np.random.random() < 0.05:
                    indicator_name = np.random.choice(list(indicators.values()))
                    row[indicator_name] = np.nan
                
                data.append(row)
        
        return pd.DataFrame(data)
    
    else:
        if indicators:
            indicators = indicators
        else:
            indicators =  {
                # Health
                'SP.DYN.LE00.IN': 'life_expectancy',
                'SP.DYN.IMRT.IN': 'infant_mortality',
                'SH.XPD.CHEX.GD.ZS': 'health_expenditure_pct_gdp',
                # Economy
                'NY.GDP.PCAP.CD': 'gdp_per_capita',
                'SI.POV.DDAY': 'poverty_2.15_usd_pct',
                'SI.POV.2DAY': 'poverty_3.65_usd_pct',
                'SI.POV.GINI': 'gini_index',
                # Education
                'SE.XPD.TOTL.GD.ZS': 'education_expenditure_pct_gdp',
                'SE.PRM.CMPT.ZS': 'primary_completion_rate',
                # Infrastructure
                'SH.STA.SMSS.ZS': 'safe_sanitation_pct',
                'EG.ELC.ACCS.ZS': 'electricity_access_pct',
                # Environment
                'EN.ATM.PM25.MC.M3': 'air_pm25',
                'EN.ATM.CO2E.PC': 'co2_per_capita',
                'AG.LND.FRST.ZS': 'forest_area_pct',
                # Demographics
                'SP.POP.TOTL': 'total_population',
                'SP.URB.GROW': 'urban_growth_pct',
                # Security
                'VC.IHR.PSRC.P5': 'homicide_rate'
            }
        if not countries:
            raise ValueError("Countries list cannot be empty.")

        # Import raw data
        raw = wb.data.DataFrame(indicators.keys(), countries, time=years, labels=False)
        raw.index.names = ['country', 'indicator']
        raw = raw.reset_index()

        # Ensure features are on the column
        df = raw.sort_values(['country','indicator'])
        df_melted = pd.melt(df, id_vars=['country', 'indicator'], var_name='year', value_name='value')
        data = df_melted.pivot(index=['country', 'year'], columns='indicator', values='value')
        data = data.reset_index()
        data['year'] = data['year'].str.replace('YR', '').astype(int)

        # Rename columns using the indicators dictionary for better readability
        data = data.rename(columns=indicators)

        data_frames = _handle_missing_values(data, missing_values_threshold, verbose)

        return data_frames


def _handle_missing_values(data: pd.DataFrame, threshold: float, verbose: bool = False) -> List[pd.DataFrame]:
    """
    Cleans the input DataFrame by handling missing values.

    Args:
        df (pd.DataFrame): The raw data DataFrame.

    Returns:
        list[pd.DataFrame]: A list of DataFrames. Each represents the processed fetched data for each given country. The final data does not contain missing values.
    """
        # Remove columns with a percentage of missing values higher to the given (or default)
        # threshold
    data_frames = []
    countries = data['country'].unique()
    for country in countries:
        curr_data = data[data['country'] == country]
        curr_missing_percentage = curr_data.isnull().sum() / len(curr_data)
        columns_to_drop = curr_missing_percentage[curr_missing_percentage > threshold].index
        if verbose:
            print(f"Columns to be removed by missing values threshold {threshold} for country {country}: {list(columns_to_drop)}")
        curr_data = curr_data.drop(columns=columns_to_drop, errors='ignore')
        data_frames.append(curr_data)
    
    for i, curr_data in enumerate(data_frames):
        df = curr_data.copy()
        group_series = df['country']
        others = df.drop(columns=['country'])
        filled_others = others.groupby(group_series, group_keys=False, sort=False).apply(lambda g: g.ffill().bfill())
        filled = filled_others.copy()
        filled['country'] = group_series
        filled = filled[df.columns]
        data_frames[i] = filled.reset_index(drop=True)
    
    return data_frames