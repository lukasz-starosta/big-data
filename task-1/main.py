import datetime
from prepare.get_prepared_data import get_prepared_data
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

queens, bronx, brooklyn, manhattan, staten = get_prepared_data()

queens.index = queens.index.rename("year", level=0)
queens.index = queens.index.rename("month", level=1)
queens = queens.rename("amount", level=0)
queens = queens.reset_index()

queens['date'] = queens.apply(lambda x:datetime.datetime.timestamp(datetime.datetime(x['year'].astype(int), x['month'].astype(int), 1)), axis=1)
final = queens[['date', 'amount']]
queens_train_x, queens_test_x, queens_train_y, queens_test_y = train_test_split(final['date'], final['amount'])

# pd.set_option("max_rows", None)
# print(queens)
#
# x=queens.index, y=queens.values

# print(final)
# print(queens_train_x.shape)
lr = LinearRegression()
lr.fit(np.array(queens_train_x).reshape(-1, 1), queens_train_y)


prediction = lr.predict(np.array(queens_test_x).reshape(-1, 1))


# final.plot(kind="bar", x='date', y='amount')
plt.plot(final['date'], final['amount'], label="Data", color='b', alpha=.5)
plt.plot(queens_test_x, prediction, label="Linear regression", color='r', alpha=.7)
plt.show()
#
# bronx.plot(kind="bar")
# plt.show()
#
# brooklyn.plot(kind="bar")
# plt.show()
#
# manhattan.plot(kind="bar")
# plt.show()
#
# staten.plot(kind="bar")
# plt.show()


