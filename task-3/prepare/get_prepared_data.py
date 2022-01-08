import pandas as pd
from os.path import exists

from prepare.map_values import get_race_number, get_sex_number
from prepare.factorize import factorize
from prepare.prepare_data import prepare_data
from sklearn.preprocessing import StandardScaler, normalize

# Relative to python invocation
DATA_PATH = './data/task-3.csv'

file_exists = exists(DATA_PATH)


def get_cleaned_data():
    if (not file_exists):
        dataframe = prepare_data()
        pd.DataFrame.to_csv(dataframe, DATA_PATH)
        return dataframe
    return pd.read_csv(DATA_PATH)


def get_prepared_data():
    data = get_cleaned_data()

    # labels
    labels = data[data.columns[6]].unique()

    # transform all records' columns values into numbers. if file exists, first column is index.
    columns = data.columns[1:] if file_exists else data.columns
    data['SUSP_RACE'] = data['SUSP_RACE'].apply(
        lambda x: get_race_number(x)
    )
    data['VIC_RACE'] = data['VIC_RACE'].apply(
        lambda x: get_race_number(x)
    )
    data['SUSP_SEX'] = data['SUSP_SEX'].apply(
        lambda x: get_sex_number(x)
    )
    data['VIC_SEX'] = data['VIC_SEX'].apply(
        lambda x: get_sex_number(x)
    )
    for col in columns:
        data[col] = factorize(data[col])

    data = data.drop(data.columns[0], axis=1)

    return data, labels

def get_normalized_data(data):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    X_normalized = normalize(X_scaled)
    return X_normalized