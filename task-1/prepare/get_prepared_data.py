import pandas as pd
from os.path import exists

from prepare.prepare_data import prepare_data

# Relative to python invocation
DATA_PATH = '../data/task-1.csv'

file_exists = exists(DATA_PATH)


def get_cleaned_data():
    if (not file_exists):
        dataframe = prepare_data()
        pd.DataFrame.to_csv(dataframe, DATA_PATH)
        return dataframe

    return pd.read_csv(DATA_PATH)


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

    date = pd.to_datetime(queens['CMPLNT_FR_DT'], format='%m/%d/%Y', errors='coerce')
    queens = queens[date.dt.year >= 2006]
    queens_grouped = queens.groupby(by=[date.dt.year, date.dt.month])

    date = pd.to_datetime(bronx['CMPLNT_FR_DT'], format='%m/%d/%Y', errors='coerce')
    bronx = bronx[date.dt.year >= 2006]
    bronx_grouped = bronx.groupby(by=[date.dt.year, date.dt.month])

    date = pd.to_datetime(brooklyn['CMPLNT_FR_DT'], format='%m/%d/%Y', errors='coerce')
    brooklyn = brooklyn[date.dt.year >= 2006]
    brooklyn_grouped = brooklyn.groupby(by=[date.dt.year, date.dt.month])

    date = pd.to_datetime(manhattan['CMPLNT_FR_DT'], format='%m/%d/%Y', errors='coerce')
    manhattan = manhattan[date.dt.year >= 2006]
    manhattan_grouped = manhattan.groupby(by=[date.dt.year, date.dt.month])

    date = pd.to_datetime(staten['CMPLNT_FR_DT'], format='%m/%d/%Y', errors='coerce')
    staten = staten[date.dt.year >= 2006]
    staten_grouped = staten.groupby(by=[date.dt.year, date.dt.month])

    return queens_grouped.size(), bronx_grouped.size(), brooklyn_grouped.size(), manhattan_grouped.size(), staten_grouped.size()
