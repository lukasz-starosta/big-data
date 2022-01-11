import datetime
import os
from prepare.get_prepared_data import get_prepared_data
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import pmdarima as pm
import pandas as pd

months = 12


def perform_regression(ds, name, degree, save):
    ds.index = ds.index.rename("year", level=0)
    ds.index = ds.index.rename("month", level=1)
    ds = ds.rename("value", level=0)
    ds = ds.reset_index()

    ds['date'] = ds.apply(
        lambda x: datetime.datetime.timestamp(datetime.datetime(x['year'].astype(int), x['month'].astype(int), 1)),
        axis=1)
    final = ds[['date', 'value']]

    ds_train_x = final['date'][:-months]
    ds_test_x = final['date'][-months:]
    ds_train_y = final['value'][:-months]
    ds_test_y = final['value'][-months:]

    polyreg_scaled = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    polyreg_scaled.fit(np.array(ds_train_x).reshape(-1, 1), ds_train_y)

    prediction = polyreg_scaled.predict(np.array(ds_test_x).reshape(-1, 1))
    x, y = zip(*sorted(zip(ds_test_x, prediction), key=lambda q: q[0]))

    mse = np.sqrt(mean_squared_error(ds_test_y, prediction))

    if save:
        path = "results/regression/"

        if not os.path.exists(path):
            os.makedirs(path)

        plt.plot(final['date'], final['value'], label="Data", color='b', alpha=.5)
        plt.plot(x, y, label="Prediction", color='r', alpha=.7)
        plt.legend()
        plt.title(name)
        plt.savefig(path + "/" + name + "_" + str(degree) + "_" + str(mse) + ".jpg")
        plt.clf()

    return mse


def regression():
    queens, bronx, brooklyn, manhattan, staten = get_prepared_data()

    # Polynomial regression
    queens_mse, bronx_mse, brooklyn_mse, manhattan_mse, staten_mse = [], [], [], [], []
    max_degree = 20
    for degree in range(1, max_degree + 1):
        print("Degree ", degree, " of ", max_degree)
        queens_mse.append(perform_regression(queens, "Queens", degree, False))
        bronx_mse.append(perform_regression(bronx, "Bronx", degree, False))
        brooklyn_mse.append(perform_regression(brooklyn, "Brooklyn", degree, False))
        manhattan_mse.append(perform_regression(manhattan, "Manhattan", degree, False))
        staten_mse.append(perform_regression(staten, "Staten_Island", degree, False))

    perform_regression(queens, "Queens", min(range(len(queens_mse)), key=queens_mse.__getitem__) + 1, True)
    perform_regression(bronx, "Bronx", min(range(len(bronx_mse)), key=bronx_mse.__getitem__) + 1, True)
    perform_regression(brooklyn, "Brooklyn", min(range(len(brooklyn_mse)), key=brooklyn_mse.__getitem__) + 1, True)
    perform_regression(manhattan, "Manhattan", min(range(len(manhattan_mse)), key=manhattan_mse.__getitem__) + 1, True)
    perform_regression(staten, "Staten_Island", min(range(len(staten_mse)), key=staten_mse.__getitem__) + 1, True)


def perform_arima(ds, name):
    path = "results/arima/"

    ds.index = ds.index.rename("year", level=0)
    ds.index = ds.index.rename("month", level=1)
    ds = ds.rename("value", level=0)
    ds = ds.reset_index()

    ds['date'] = ds.apply(
        lambda x: datetime.datetime(x['year'].astype(int), x['month'].astype(int), 1), axis=1)
    final = ds[['date', 'value']]
    final = final.set_index('date')

    ds_train_x = final[:-months]

    model = pm.auto_arima(ds_train_x["value"], trace=True,
                          suppress_warnings=True, seasonal=True)

    # Forecast
    fitted, confint = model.predict(n_periods=2 * months, return_conf_int=True)
    index_of_fc = pd.date_range(ds_train_x.index[-1], periods=2 * months, freq='MS')

    # make series for plotting purpose
    fitted_series = pd.Series(fitted, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)

    # Plot
    plt.plot(final.index, final['value'], label="Data", color='b', alpha=.5)
    plt.plot(fitted_series, color='darkgreen')
    plt.fill_between(lower_series.index,
                     lower_series,
                     upper_series,
                     color='k', alpha=.15)
    plt.title(name)

    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(path + "/" + name + "_" + str(model.aic()) + ".jpg")
    plt.clf()


def arima():
    queens, bronx, brooklyn, manhattan, staten = get_prepared_data()
    perform_arima(queens, "Queens")
    perform_arima(bronx, "Bronx")
    perform_arima(brooklyn, "Brooklyn")
    perform_arima(manhattan, "Manhattan")
    perform_arima(staten, "Staten Island")
