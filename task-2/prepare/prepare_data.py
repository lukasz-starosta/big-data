import numpy as np
import pandas as pd
from pandas.core.algorithms import factorize

MAIN_DATAFILE_PATH = '../data/NYPD_Complaint_Data_Historic.csv'


def prepare_data():
    initial_data = pd.read_csv(
        MAIN_DATAFILE_PATH, na_values=np.nan)

    # get only selected columns from the dataset
    data = initial_data[['LAW_CAT_CD', 'BORO_NM', 'PREM_TYP_DESC',
                         'SUSP_AGE_GROUP', 'SUSP_RACE', 'VIC_AGE_GROUP', 'VIC_RACE', 'VIC_SEX', 'SUSP_SEX']]

    # delete records which do not have suspect's sex defined - they provide no value
    data = data[data['SUSP_SEX'].isin([
        'M', 'F'])]

    # replace UNKNOWN for np.nan
    data.replace('UNKNOWN', np.nan, inplace=True)

    # delete records with unknown/no value
    data.dropna(subset=['BORO_NM', 'VIC_RACE', 'PREM_TYP_DESC', 'VIC_AGE_GROUP',
                'VIC_SEX', 'SUSP_AGE_GROUP', 'SUSP_RACE'], inplace=True)

    data = data[data['SUSP_AGE_GROUP'].isin([
        '<18', '18-24', '25-44', '45-64', '65+'])]

    data = data[data['VIC_AGE_GROUP'].isin([
        '<18', '18-24', '25-44', '45-64', '65+'])]

    return data
