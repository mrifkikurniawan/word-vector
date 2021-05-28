from sklearn.manifold import TSNE                   # final reduction
import numpy as np                                  # array handling


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


def plot_with_plotly(x_vals: np.ndarray, y_vals: np.ndarray, labels: np.ndarray, cluster: np.ndarray,
                     plot_in_notebook=True):
    
    from plotly.offline import init_notebook_mode, iplot, plot
    import plotly.graph_objs as go

    trace = go.Scatter(x=x_vals, 
                       y=y_vals, 
                       mode='text', 
                       text=labels,
                       marker=dict(color=cluster))
    data = [trace]

    if plot_in_notebook:
        init_notebook_mode(connected=True)
        iplot(data, filename='word-embedding-plot')
    else:
        plot(data, filename='word-embedding-plot.html')


def plot_with_matplotlib(x_vals: np.ndarray, y_vals: np.ndarray, labels: np.ndarray, 
                         cluster: np.ndarray):
    import matplotlib.pyplot as plt
    import random

    random.seed(0)

    plt.figure(figsize=(25, 25))
    plt.scatter(x_vals, y_vals, c=cluster)

    #
    # Label randomly subsampled 25 data points
    #
    indices = list(range(len(labels)))
    selected_indices = random.sample(indices, 25)
    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))