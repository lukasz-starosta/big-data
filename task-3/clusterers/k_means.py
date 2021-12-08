import sys
from sklearn.cluster import KMeans

from clusterers.clusterer import Clusterer


class Kmeans(Clusterer, KMeans):
    name = "K-means"

    def __init__(self, data, n_clusters):
        super().__init__(data)

        n_clusters = n_clusters

        n_init = 1
        n_init = n_init if n_init > 0 else 10

        max_iter = 30
        max_iter = max_iter if max_iter > 0 else 300
        
        try:
            random_state = 1
            random_state = random_state if random_state > 0 else None
        except:
            random_state = None

        self.model = KMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state
        )


    def get_inertia_for_elbow(self):
        self.model.fit(self.data)
        return self.model.inertia_
