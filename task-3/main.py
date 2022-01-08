from matplotlib import pyplot
from prepare.get_prepared_data import get_prepared_data, get_normalized_data
import pandas as pd
from clusterers.k_means import Kmeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from utils.functions import get_rand_records, display_factorial_planes, display_parallel_coordinates, display_parallel_coordinates_centroids
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

data, labels = get_prepared_data()
pca = PCA(n_components=2)

data_sample_arr = []
data_sample_norm_arr = []
data_labels_arr = []
silhouettes_arr = []
cluster_centres_arr = []
silhouette_sum = 0
CHS_sum = 0
DBS_sum = 0
silhouette_mean = 0
CHS_mean = 0
DBS_mean = 0
iterations = 20

for i in range(iterations):
    temp_data_sample = get_rand_records(data, 5000)
    temp_data_sample_norm = get_normalized_data(temp_data_sample)
    data_sample_arr.append(temp_data_sample)
    data_sample_norm_arr.append(temp_data_sample_norm)
    kmeans = KMeans(init='random', n_clusters=4, n_init=10)
    kmeans.fit(temp_data_sample_norm)
    cluster_centres_arr.append(kmeans.cluster_centers_)
    temp_data_labels = kmeans.predict(temp_data_sample_norm)
    data_labels_arr.append(temp_data_labels)
    temp_silhouette_score = silhouette_score(temp_data_sample_norm, temp_data_labels)
    silhouette_sum += temp_silhouette_score
    silhouettes_arr.append(temp_silhouette_score)

silhouette_mean = silhouette_sum / iterations


data_sample = 0
data_sample_norm = 0
data_labels = 0
cluster_centres = 0
best_silhouette_mean = 0
min_abs = 1
best_index = 0
for i in range(iterations):
    if(abs(silhouettes_arr[i] - silhouette_mean) < min_abs):
        min_abs = abs(silhouettes_arr[i] - silhouette_mean)
        best_silhouette_mean = silhouettes_arr[i]
        data_sample = data_sample_arr[i]
        data_sample_norm = data_sample_norm_arr[i]
        data_labels = data_labels_arr[i]
        cluster_centres = cluster_centres_arr[i]
        best_index = i

X_principal = pca.fit_transform(data_sample_norm)
X_principal_df = pd.DataFrame(X_principal, index=data_sample.index, columns=['PC1', 'PC2'])
centres_reduced = pca.transform(cluster_centres)

display_factorial_planes(X_principal, 2, pca, [(0,1)], illustrative_var = data_labels, alpha = 0.8)
pyplot.scatter(centres_reduced[:, 0], centres_reduced[:, 1],
            marker='x', s=169, linewidths=3,
            color='b', zorder=10)

X_clustered = pd.DataFrame(data_sample_norm, index=data_sample.index, columns=data_sample.columns)
X_clustered["cluster"] = data_labels

# Display parallel coordinates plots, one for each cluster
display_parallel_coordinates(X_clustered, 4)
pyplot.show()

centroids = pd.DataFrame(cluster_centres, columns=data_sample.columns)
centroids['cluster'] = centroids.index

display_parallel_coordinates_centroids(centroids, 4)
pyplot.show()


elbow_points = []
K = range(1, 10)
for k in K:
    kmeans = Kmeans(data_sample, k)
    elbow_points.append(kmeans.get_inertia_for_elbow())

pyplot.plot(K, elbow_points, 'bx-')
pyplot.xlabel('k')
pyplot.ylabel('Distortion')
pyplot.show()
