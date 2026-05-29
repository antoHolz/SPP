import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering, HDBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics.pairwise import cosine_distances


class ClusterModel:
    """Wraps a fitted clusterer so it can predict on new BxD points."""
    def __init__(self, predict_fn, n_clusters):
        self._predict = predict_fn
        self.n_clusters = n_clusters

    def predict(self, X):
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        return self._predict(X)

    def __call__(self, X):
        return self.predict(X)


def cluster_tensors(X, mode="kMeans", n_clusters=10, metric="cosine",
                    linkage="average", weights=None, k=5, random_state=0):
    """
    Cluster a BxD tensor/array with 'kMeans', 'HDBSCAN', or 'ACsilhouette'.
    Returns a ClusterModel that assigns new BxD points to clusters.

    For AC/HDBSCAN, new points are assigned by k-NN vote over the fitted
    points (k=1 = nearest training point). HDBSCAN additionally uses its
    own approximate_predict to preserve noise labels.
    """
    if isinstance(X, torch.Tensor):
        X_np = X.detach().cpu().numpy()
    else:
        X_np = np.asarray(X)

    if mode == "kMeans":
        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state,
                             batch_size=64, n_init="auto").fit(X_np)
        return ClusterModel(lambda Y: km.predict(Y), n_clusters)

    if mode == "HDBSCAN":
        hdb = HDBSCAN(min_cluster_size=5, metric=metric).fit(X_np)
        labels = hdb.labels_
        n_found = int(labels.max() + 1) if (labels >= 0).any() else 0
        if n_found == 0:
            raise RuntimeError("HDBSCAN assigned all points to noise.")
        # kNN vote over the non-noise fitted points
        valid = labels >= 0
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric).fit(X_np[valid], labels[valid])
        return ClusterModel(lambda Y: knn.predict(Y), n_found)

    if mode == "ACsilhouette":
        candidates = np.arange(2, max(n_clusters, 11))
        best_score, best_labels = -np.inf, None
        for n in candidates:
            lbl = AgglomerativeClustering(n_clusters=n, metric=metric,
                                          linkage=linkage).fit_predict(X_np)
            if weights is not None:
                s = np.average(silhouette_samples(X_np, lbl, metric=metric), weights=weights)
            else:
                s = silhouette_score(X_np, lbl, metric=metric)
            if s > best_score:
                best_score, best_labels = s, lbl
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric).fit(X_np, best_labels)
        return ClusterModel(lambda Y: knn.predict(Y), int(best_labels.max() + 1))

    raise ValueError(f"Unknown mode {mode!r}.")


def silhouette(X, labels, metric="cosine"):
    """
    Scalar silhouette score for a clustering. Thin wrapper around
    `sklearn.metrics.silhouette_score` with two guards:
      - returns NaN if fewer than 2 unique non-noise labels (score undefined),
      - converts torch tensors to numpy.

    Default metric matches `cluster_tensors` (cosine).
    """
    if isinstance(X, torch.Tensor):
        X_np = X.detach().cpu().numpy()
    else:
        X_np = np.asarray(X)
    lbl = np.asarray(labels)

    # 1) sklearn requires >= 2 distinct labels and any noise points (-1) skew
    #    the score; drop them before counting unique classes
    mask = lbl >= 0 if (lbl < 0).any() else np.ones_like(lbl, dtype=bool)
    if np.unique(lbl[mask]).size < 2:
        return float("nan")
    return float(silhouette_score(X_np[mask], lbl[mask], metric=metric))