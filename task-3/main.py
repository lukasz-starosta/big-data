from prepare.get_prepared_data import get_prepared_data
import pandas as pd
from clusterers.agglomerative import Agglomerative
from clusterers.k_means import Kmeans
from sklearn.metrics import silhouette_score


data, labels = get_prepared_data()
algorithm = Kmeans(data)
data_labels = algorithm.fit_predict()
print(f'Silhouette:\t\t%0.4f' % silhouette_score(data, data_labels))
print(len(data.index))