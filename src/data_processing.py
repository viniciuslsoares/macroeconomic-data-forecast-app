import pandas as pd
import numpy as np
from typing import List

# Modo de desenvolvimento: usar dados fictícios para testar o frontend
DEVELOPMENT_MODE = True

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
    if DEVELOPMENT_MODE:
        # Gerar dados fictícios realistas para desenvolvimento do frontend
        np.random.seed(42)  # Para resultados consistentes
        
        data = []
        years = list(range(start_year, end_year + 1))
        
        for country in countries:
            for year in years:
                row = {'economy': country, 'Year': year}
                
                # Gerar dados realistas baseados no país
                if country == 'BRA':  # Brasil
                    row['GDP (current US$)'] = 1.8e12 + (year - 2000) * 5e10 + np.random.normal(0, 1e11)
                    row['Population, total'] = 200e6 + (year - 2000) * 1.5e6 + np.random.normal(0, 1e6)
                    row['CO2 emissions (kt)'] = 400000 + (year - 2000) * 5000 + np.random.normal(0, 20000)
                    row['Life expectancy at birth, total (years)'] = 70 + (year - 2000) * 0.2 + np.random.normal(0, 0.5)
                    row['Individuals using the Internet (% of population)'] = min(95, 10 + (year - 2000) * 3.5 + np.random.normal(0, 2))
                
                elif country == 'CAN':  # Canadá
                    row['GDP (current US$)'] = 1.2e12 + (year - 2000) * 3e10 + np.random.normal(0, 5e10)
                    row['Population, total'] = 31e6 + (year - 2000) * 0.8e6 + np.random.normal(0, 0.5e6)
                    row['CO2 emissions (kt)'] = 550000 + (year - 2000) * -2000 + np.random.normal(0, 15000)
                    row['Life expectancy at birth, total (years)'] = 79 + (year - 2000) * 0.1 + np.random.normal(0, 0.3)
                    row['Individuals using the Internet (% of population)'] = min(98, 50 + (year - 2000) * 2 + np.random.normal(0, 1))
                
                # Adicionar alguns valores NaN ocasionais para testar o preprocessing
                if np.random.random() < 0.05:  # 5% chance de NaN
                    indicator_name = np.random.choice(list(indicators.values()))
                    row[indicator_name] = np.nan
                
                data.append(row)
        
        return pd.DataFrame(data)
    
    else:
        # Código original para produção (comentado para evitar erros)
        # import wbgapi as wb
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
    # Implementação funcional para desenvolvimento
    df_cleaned = df.groupby('economy').apply(
        lambda group: group.interpolate(method='linear', limit_direction='forward', axis=0)
    ).reset_index(drop=True)
    
    return df_cleaned