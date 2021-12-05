from sklearn.cluster import AgglomerativeClustering

from clusterers.clusterer import Clusterer


class Agglomerative(Clusterer):
    name = "Agglomerative hierarchical clustering"

    def __init__(self, data, n_clusters):
        super().__init__(data)
        try:
            linkage = 'single'    # ward, complete, average, single
            affinity = 'manhattan'    # euclidean, manhattan
            self.model = AgglomerativeClustering(
                n_clusters=5,
                linkage=linkage,
                affinity=affinity
            )
        except:
            self.model = AgglomerativeClustering(
                n_clusters=n_clusters
            )
