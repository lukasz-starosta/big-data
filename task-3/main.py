import time

import numpy as np
from matplotlib import pyplot
from prepare.get_prepared_data import get_prepared_data
import pandas as pd
from clusterers.agglomerative import Agglomerative
import tensorflow as tf
from tensorflow.python.client import device_lib as dev_lib
#from kmeanstf import KMeansTF
from clusterers.k_means import Kmeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from utils.functions import get_rand_records
from sklearn.cluster import KMeans

data, labels = get_prepared_data()

data_sample_arr = []
data_labels_arr = []
silhouettes_arr = []
silhouette_sum = 0
CHS_sum = 0
DBS_sum = 0
silhouette_mean = 0
CHS_mean = 0
DBS_mean = 0
iterations = 20

for i in range(iterations):
    temp_data_sample = get_rand_records(data, 10000)
    data_sample_arr.append(temp_data_sample)
    algorithm = Kmeans(temp_data_sample, 7)
    #algorithm = KMeans(n_clusters = 10).fit(temp_data_sample)
    #temp_data_labels = algorithm.labels_
    temp_data_labels = algorithm.fit_predict()
    data_labels_arr.append(temp_data_labels)
    temp_silhouette_score = silhouette_score(temp_data_sample, temp_data_labels)
    silhouette_sum += temp_silhouette_score
    silhouettes_arr.append(temp_silhouette_score)
    #CHS_sum += calinski_harabasz_score(temp_data_sample, temp_data_labels)
    #DBS_sum += davies_bouldin_score(temp_data_sample, temp_data_labels)

silhouette_mean = silhouette_sum / iterations

print(silhouette_mean)


data_sample = 0
data_labels = 0
best_silhouette_mean = 0
min_abs = 1
best_index = 0
for i in range(iterations):
    if(abs(silhouettes_arr[i] - silhouette_mean) < min_abs):
        min_abs = abs(silhouettes_arr[i] - silhouette_mean)
        best_silhouette_mean = silhouettes_arr[i]
        data_sample = data_sample_arr[i]
        data_labels = data_labels_arr[i]
        best_index = i
    print(silhouettes_arr[i], abs(silhouettes_arr[i] - silhouette_mean), best_silhouette_mean)
print(best_silhouette_mean, calinski_harabasz_score(data_sample, data_labels), davies_bouldin_score(data_sample, data_labels), best_index)

elbow_points = []
K = range(1, 10)
for k in K:
    kmeans = Kmeans(data_sample, k)
    elbow_points.append(kmeans.get_inertia_for_elbow())

pyplot.plot(K, elbow_points, 'bx-')
pyplot.xlabel('k')
pyplot.ylabel('Distortion')
pyplot.show()


'''x = data_sample.iloc[:, 6]
xlabel = data_sample.columns[6]
y = data_sample.iloc[:, 1]
ylabel = data_sample.columns[1]

print(xlabel, ylabel)

pyplot.scatter(
        x=x,
        y=y,
        c=data_labels,
        cmap='rainbow'
    )

pyplot.xlabel(xlabel)
pyplot.ylabel(ylabel)
pyplot.savefig("data/clustering", dpi=200, bbox_inches='tight')
pyplot.show()'''

'''print(f'Silhouette:', silhouette_score(data_sample, data_labels))
print(f'Calinski Harabasz score:', calinski_harabasz_score(data_sample, data_labels))
print(f'Davies Bouldin score:', davies_bouldin_score(data_sample, data_labels))'''


'''config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.Session(config=config, ...)'''
'''print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
  print(tf.config.experimental.get_memory_info('GPU:0')['current'])'''
'''
config = tf.compat.v1.ConfigProto()
start_time = time.time()
kmeanstf = KMeansTF(n_clusters=10)
data, labels = get_prepared_data()
data_test = data[:100000]
data_labels = kmeanstf.fit(data_test)
end_time = time.time()
print('Kmeans++ execution time in seconds: {}'.format(end_time - start_time))'''
'''
algorithm = Kmeans(data_test)
print(algorithm.get_inertia_for_elbow())
data_labels = algorithm.fit_predict()
print(f'Silhouette:\t\t%0.4f' % silhouette_score(data_test, data_labels))
print(f'Calinski Harabasz score:\t\t%0.4f' % calinski_harabasz_score(data_test, data_labels))
print(f'Davies Bouldin score:\t\t%0.4f' % davies_bouldin_score(data_test, data_labels))
print(len(data.index))
'''