import numpy as np
import pandas as pd
from pandas.core.algorithms import factorize

MAIN_DATAFILE_PATH = '../data/NYPD_Complaint_Data_Historic.csv'


def prepare_data():
    initial_data = pd.read_csv(
        MAIN_DATAFILE_PATH, na_values=np.nan)

    # get only selected columns from the dataset
    data = initial_data[['CMPLNT_FR_DT', 'BORO_NM', 'OFNS_DESC']]

    # replace UNKNOWN for np.nan
    data.replace('UNKNOWN', np.nan, inplace=True)

    # delete records with unknown/no value
    data.dropna(subset=['CMPLNT_FR_DT', 'BORO_NM', 'OFNS_DESC'], inplace=True)

    data = data[data['OFNS_DESC'].isin([
        'PETIT LARCENY'])]

    return data
