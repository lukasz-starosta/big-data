import pandas as pd
from os.path import exists
from prepare.prepare_data import prepare_data

# Relative to python invocation
DATA_PATH = './data/task-2.csv'

file_exists = exists(DATA_PATH)


def get_prepared_data():
    if (not file_exists):
        dataframe = prepare_data()
        pd.DataFrame.to_csv(dataframe, DATA_PATH)
        return dataframe

    return pd.read_csv(DATA_PATH)
