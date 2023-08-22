# GapStatistics Package

The GapStatistics package provides a Python implementation of the Gap Statistics method for determining the optimal number of clusters in a dataset using K-means clustering derived from [Tibshirani et al.](https://hastie.su.domains/Papers/gap.pdf). The package is designed to choose the distance metrics agnostically as well as the clustering algorithm. However, the primary usage is the KMeans algorithm. 

## Features

- Calculate the Gap Statistics for determining the optimal number of clusters in a dataset.
- Supports various distance metrics for clustering.
- Provides options for applying Principal Component Analysis (PCA) during bootstrapping.
- Allows to choose whether to return additional statistics for analysis.

## Installation

To install the GapStatistics package, you can use pip:

```bash
pip install gapstatistics
```

## Example - Training / Prediction

This is the basic use case. If you don't define the parameter <em>algorithm</em> parameter, the default clustering technique is <em>KMeans</em>. The returned object <em>optimum</em> is an **integer** showing the optimal number of clusters for the data.

```
from gapstatistics import GapStatistics

centers = [[0,0], [0,6], [3,2], [5,0]]
X = make_blobs(n_samples=200, centers=centers, n_features=2, cluster_std=1)
n_iterations = 30

gs = GapStatistics(distance_metric='minkowski')

optimum = Gs.fit_predict(K=10, X=X[0])

print(f'Optimum: {optimum}')
```

## Example - Visualization

Here is some code that you can use for showing different plots how the gap statistics derives the optimal number of clusters. For this, you must set the *return_params* to **True**, so that you can plot them. 

```
from gapstatistics import GapStatistics
from sklearn.datasets import make_blobs

centers = [[0,0], [0,6], [3,2], [5,0]]
X = make_blobs(n_samples=200, centers=centers, n_features=2, cluster_std=1)
n_iterations = 30

gs = GapStatistics(distance_metric='minkowski', return_params=True)

optimum, params = gs.fit_predict(K=10, X=X[0])

gs.plot()

```

## Example - Provide custom distance metrics

Here is some code that you can use for creating a custom distance metric to provide to the class. The distance metric must have two parameters:

- X: (list)
- Centroid: (list)

```
def euclidian_distance(X: np.array, Centroid: np.array) -> np.array:
    return np.linalg.norm(X - Centroid, axis=1)

def manhattan_distance(X: np.array, Centroid: np.array) -> np.array:
    return np.sum(np.abs(X - Centroid), axis=1)

GapStatistics(distance_metric=manhattan_distance)
```

## Parameters

### GapStatistics Class

#### `__init__(self, algorithm, distance_metric, pca_sampling, return_params)`

- `algorithm` (Callable): The clustering algorithm to use (default: KMeans). It should be a callable that creates a clustering model. If you want to use your own clustering algorithm, you must provide a callable object that has the following attributes / functions:
  - __init__(n_clusters)
  - fit
  - predict
  - self.cluster_centers_
- `distance_metric` (str or callable): The distance metric used for clustering. If a string, it should be a valid metric name recognized by `sklearn.metrics.DistanceMetric`. If a callable, it should accept two arrays and return the distance between them. Examples for strings are:
  - manhattan
  - euclidean
  - l1, l2
  - minkowski
    
- `pca_sampling` (bool): Whether to apply Principal Component Analysis (PCA) during bootstrapping (default: True).
- `return_params` (bool): Whether to return additional statistics in the `fit_predict` function (default: False).

### Methods

#### `fit_predict(self, K, X, n_iterations)`

Perform gap statistics to find the optimal number of clusters (K) for a given dataset.

- `K` (int): The maximum number of clusters (K) to consider for finding the optimal K value.
- `X` (list): A list of data points (samples) to be used for clustering. Must have a 2D shape -> (?, 2)
- `n_iterations` (int): The number of iterations to perform for simulating Wk's statistics.

Returns either the optimal number of clusters (K) or a tuple with the optimal K and additional statistics used in gap analysis.

#### `plot(self, original_labels, colors)`

Visualize the output of the gap statistics based on the returned parameters.

- `original_labels` (list): The list of the original groundtruth labels to compare against (if accessible).
- `colors` (dict): If the optimal value is greater than 10, you must provide an additional color dictionary.

Returns a plot consisting of four subplots for showing why the gap statistics decided the optimal number of clusters.
