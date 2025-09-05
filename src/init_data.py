import wbgapi as wb
import pandas as pd
import numpy as np

class WBGData():
    @classmethod
    def get_data(cls,
                countries: list[str], years: list[int], indicators: dict[str, str] = [],
                missing_values_threshold: int = 0.3, verbose: bool = False
            ) -> pd.DataFrame:
        '''
        Import data directly from wbgapi library, and simply process it to structure features on the columns.

        Parameters
        ----------
        countries: list[str]
            List of countries that will be imported. Names must follow the pattern of WBGAPI.
        years: list[int]
            List of all years that will be imported for each country.
        threshold: float
            Features with a percentage of missing values higher than the given threshold will be removed.
            As a result, each country may have different features removed. Default value is 0.3.
        verbose: bool
            If set to true, the method will print the features to be removed for each country.

        Returns
        -------
        list[pd.DataFrame]
            A list of data frames. Each element represents the cleaned data for each country.
        '''

        if indicators:
            cls.indicators = indicators
        else:
            cls.indicators =  {
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
        if countries:
            cls.countries = countries
        else:
            raise ValueError("Countries list cannot be empty.")
        if years:
            cls.years = years
        else:
            raise ValueError("Years list cannot be empty.")

        # Import raw data
        raw = wb.data.DataFrame(cls.indicators.keys(), cls.countries, time=cls.years, labels=False)
        raw.rename(columns=cls.indicators, inplace=True)
        raw.index.names = ['country', 'indicator']
        raw = raw.reset_index()

        # Ensure features are on the column
        df = raw.sort_values(['country','indicator'])
        df_melted = pd.melt(df, id_vars=['country', 'indicator'], var_name='year', value_name='value')
        cls.data = df_melted.pivot(index=['country', 'year'], columns='indicator', values='value')
        cls.data = cls.data.reset_index()
        cls.data['year'] = cls.data['year'].str.replace('YR', '').astype(int)

        data_frames = cls._handle_missing_values(missing_values_threshold, verbose)

        return data_frames
    
    @classmethod
    def _handle_missing_values(cls, threshold: int, verbose: bool = False) -> list[pd.DataFrame]:
        '''
        Handle missing values for each country on the instatiated data.
        
        Parameters
        ----------
        threshold: float
            Features with a percentage of missing values higher than the given threshold will be removed.
            As a result, each country may have different features removed. Default value is 0.3.
        verbose: bool
            If set to true, the method will print the features to be removed for each country.
        
        Returns
        -------
        list[pd.DataFrame]
            A list of data frames. Each element represents the cleaned data for each country.
        '''
        # Remove columns with a percentage of missing values higher to the given (or default)
        # threshold
        data_frames = []
        for country in cls.countries:
            curr_data = cls.data[cls.data['country'] == country]
            curr_missing_percentage = curr_data.isnull().sum() / len(curr_data)
            columns_to_drop = curr_missing_percentage[curr_missing_percentage > threshold].index
            if verbose:
                print(f"Columns to be removed by missing values threshold {threshold} for country {country}: {list(columns_to_drop)}")
            curr_data = curr_data.drop(columns=columns_to_drop, errors='ignore')
            data_frames.append(curr_data)
        

        # Fill the missing values for each country
        for i, curr_data in enumerate(data_frames):
            # data_frames[i] = curr_data.groupby('country', group_keys=False).apply(lambda g: g.ffill().bfill()).reset_index(drop=True)
            df = curr_data.copy()
            # separate the grouping series and the rest of the columns
            group_series = df['country']
            others = df.drop(columns=['country'])
            # apply ffill/bfill to the non-grouping columns grouped by the series
            filled_others = others.groupby(group_series, group_keys=False, sort=False).apply(lambda g: g.ffill().bfill())
            # reattach the country column and restore original column order
            filled = filled_others.copy()
            filled['country'] = group_series
            filled = filled[df.columns]
            # match the original behaviour which reset the index
            data_frames[i] = filled.reset_index(drop=True)
        
        return data_frames