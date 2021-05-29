from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE 
import numpy as np

def cluster_dbscan(vectors: np.ndarray, **kwargs):
    clustering = DBSCAN(**kwargs)
    clustering.fit(vectors)

    return clustering.labels_


def reduce_dimensions(vectors: np.ndarray):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    # extract the words & their vectors, as numpy arrays
    vectors = np.asarray(vectors)

    # reduce using t-SNE
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals