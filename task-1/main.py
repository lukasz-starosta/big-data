import datetime
import os
from prepare.get_prepared_data import get_prepared_data
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

months = 12


# degree = 8


def perform_regression(ds, name):
    path = "results/" + str(degree)

    ds.index = ds.index.rename("year", level=0)
    ds.index = ds.index.rename("month", level=1)
    ds = ds.rename("amount", level=0)
    ds = ds.reset_index()

    ds['date'] = ds.apply(
        lambda x: datetime.datetime.timestamp(datetime.datetime(x['year'].astype(int), x['month'].astype(int), 1)),
        axis=1)
    final = ds[['date', 'amount']]

    # queens_train_x, queens_test_x, queens_train_y, queens_test_y = train_test_split(final['date'], final['amount'])

    ds_train_x = final['date'][:-months]
    ds_test_x = final['date'][-months:]
    ds_train_y = final['amount'][:-months]
    ds_test_y = final['amount'][-months:]

    # scaler = StandardScaler()
    polyreg_scaled = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    polyreg_scaled.fit(np.array(ds_train_x).reshape(-1, 1), ds_train_y)

    prediction = polyreg_scaled.predict(np.array(ds_test_x).reshape(-1, 1))
    x, y = zip(*sorted(zip(ds_test_x, prediction), key=lambda q: q[0]))

    mse = np.sqrt(mean_squared_error(ds_test_y, prediction))

    plt.plot(ds_train_x, ds_train_y, label="Data", color='b', alpha=.5)
    plt.plot(x, y, label="Prediction", color='r', alpha=.7)
    plt.legend()

    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(path + "/" + name + "_" + str(mse) + ".jpg")
    plt.clf()
    return mse


queens, bronx, brooklyn, manhattan, staten = get_prepared_data()

queens_mse, bronx_mse, brooklyn_mse, manhattan_mse, staten_mse = [], [], [], [], []
max_degree = 20
for degree in range(1, max_degree + 1):
    print("Degree ", degree, " of ", max_degree)
    queens_mse.append(perform_regression(queens, "queens"))
    bronx_mse.append(perform_regression(bronx, "bronx"))
    brooklyn_mse.append(perform_regression(brooklyn, "brooklyn"))
    manhattan_mse.append(perform_regression(manhattan, "manhattan"))
    staten_mse.append(perform_regression(staten, "staten"))

print()
print("Queens: ", min(range(len(queens_mse)), key=queens_mse.__getitem__) + 1)
print("Bronx: ", min(range(len(bronx_mse)), key=bronx_mse.__getitem__) + 1)
print("Brooklyn: ", min(range(len(brooklyn_mse)), key=brooklyn_mse.__getitem__) + 1)
print("Manhattan: ", min(range(len(manhattan_mse)), key=manhattan_mse.__getitem__) + 1)
print("Staten Island: ", min(range(len(staten_mse)), key=staten_mse.__getitem__) + 1)
