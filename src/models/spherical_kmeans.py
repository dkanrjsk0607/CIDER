# src/models/spherical_kmeans.py
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state


class SphericalKMeans(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=8, max_iter=300, n_init=10, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

    def _normalize(self, X):
        norms = np.sqrt(np.sum(X**2, axis=1))
        norms[norms == 0] = 1.0
        return X / norms[:, np.newaxis]

    def _init_centroids(self, X, random_state):
        n_samples = X.shape[0]
        random_indices = random_state.permutation(n_samples)[: self.n_clusters]
        centroids = X[random_indices]
        return self._normalize(centroids)

    def fit(self, X, y=None):
        random_state = check_random_state(self.random_state)
        X = np.asarray(X)
        X = self._normalize(X)

        best_inertia = None
        best_labels = None
        best_centers = None

        for i in range(self.n_init):
            centers = self._init_centroids(X, random_state)
            for j in range(self.max_iter):
                cosine_sim = np.dot(X, centers.T)
                labels = np.argmax(cosine_sim, axis=1)

                new_centers = np.zeros_like(centers)
                for k in range(self.n_clusters):
                    points_in_cluster = X[labels == k]
                    if len(points_in_cluster) > 0:
                        new_centers[k] = points_in_cluster.mean(axis=0)

                new_centers = self._normalize(new_centers)

                center_shift = np.sum((centers - new_centers) ** 2)
                centers = new_centers
                if center_shift < 1e-4:
                    break

            cosine_sim = np.dot(X, centers.T)
            nearest_cosine = np.max(cosine_sim, axis=1)
            inertia = -np.sum(nearest_cosine)
            if best_inertia is None or inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
                best_centers = centers

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        return self

    def predict(self, X):
        X = np.asarray(X)
        X = self._normalize(X)
        cosine_sim = np.dot(X, self.cluster_centers_.T)
        return np.argmax(cosine_sim, axis=1)

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.labels_
