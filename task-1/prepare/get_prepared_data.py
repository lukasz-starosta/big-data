import pandas as pd
from os.path import exists

from prepare.prepare_data import prepare_data

# Relative to python invocation
DATA_PATH = '../data/task-1.csv'

file_exists = exists(DATA_PATH)


def get_cleaned_data():
    if not file_exists:
        dataframe = prepare_data()
        pd.DataFrame.to_csv(dataframe, DATA_PATH)
        return dataframe

    return pd.read_csv(DATA_PATH)


def group_by_dates(ds):
    date = pd.to_datetime(ds['CMPLNT_FR_DT'], format='%m/%d/%Y', errors='coerce')
    ds = ds[date.dt.year >= 2006]
    return ds.groupby(by=[date.dt.year, date.dt.month]).size()


def get_prepared_data():
    data = get_cleaned_data()

    queens = data[data['BORO_NM'].isin([
        'QUEENS'])]
    bronx = data[data['BORO_NM'].isin([
        'BRONX'])]
    brooklyn = data[data['BORO_NM'].isin([
        'BROOKLYN'])]
    manhattan = data[data['BORO_NM'].isin([
        'MANHATTAN'])]
    staten = data[data['BORO_NM'].isin([
        'STATEN ISLAND'])]

    return group_by_dates(queens), group_by_dates(bronx), group_by_dates(brooklyn), group_by_dates(
        manhattan), group_by_dates(staten)
