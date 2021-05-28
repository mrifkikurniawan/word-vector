from sklearn.cluster import DBSCAN
import numpy as np

def cluster_dbscan(vectors: np.ndarray, **kwargs):
    clustering = DBSCAN(**kwargs)
    clustering.fit(vectors)

    return clustering.labels_