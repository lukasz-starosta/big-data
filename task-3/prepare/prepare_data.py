import numpy as np
import pandas as pd
from pandas.core.algorithms import factorize

MAIN_DATAFILE_PATH = '../data/NYPD_Complaint_Data_Historic.csv'


def prepare_data():
    initial_data = pd.read_csv(
        MAIN_DATAFILE_PATH, na_values=np.nan)


    # get only selected columns from the dataset
    data = initial_data[['OFNS_DESC', 'BORO_NM', 'LOC_OF_OCCUR_DESC',
                         'PREM_TYP_DESC', 'SUSP_AGE_GROUP', 'SUSP_RACE', 'SUSP_SEX',
                         'VIC_AGE_GROUP', 'VIC_RACE', 'VIC_SEX']]

    data = data[data['SUSP_SEX'].isin([
        'M', 'F'])]

    # replace UNKNOWN for np.nan
    data.replace('UNKNOWN', np.nan, inplace=True)

    # delete records with unknown/no value
    data.dropna(subset=['OFNS_DESC', 'BORO_NM', 'LOC_OF_OCCUR_DESC',
                         'PREM_TYP_DESC', 'SUSP_AGE_GROUP', 'SUSP_RACE', 'SUSP_SEX',
                         'VIC_AGE_GROUP', 'VIC_RACE', 'VIC_SEX'], inplace=True)

    data = data[data['SUSP_AGE_GROUP'].isin([
        '<18', '18-24', '25-44', '45-64', '65+'])]

    data = data[data['VIC_AGE_GROUP'].isin([
        '<18', '18-24', '25-44', '45-64', '65+'])]


    return data