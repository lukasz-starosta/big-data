import numpy
import pandas


def factorize(column):
    if column.dtype in [numpy.float64, numpy.float32, numpy.int32, numpy.int64]:
        return column
    else:
        return pandas.factorize(column)[0]
