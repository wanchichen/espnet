#!/usr/bin/python

import os
import pickle

import numpy as np
import threadpoolctl


def threadpool_limits(limits=None, user_api=None):
    if not hasattr(threadpoolctl, "ThreadpoolController"):
        controller = None
    else:
        controller = threadpoolctl.ThreadpoolController()

    if controller is not None:
        return controller.limit(limits=limits, user_api=user_api)
    else:
        return threadpoolctl.threadpool_limits(limits=limits, user_api=user_api)


def _labels_inertia_threadpool_limit(
    X, sample_weight, centers, n_threads=1, return_inertia=True
):
    with threadpool_limits(limits=1, user_api="blas"):
        result = _labels_inertia(X, sample_weight, centers, n_threads, return_inertia)
    return result


def _labels_inertia(X, sample_weight, centers, n_threads=1, return_inertia=True):
    """E step of the K-means EM algorithm.

    Compute the labels and the inertia of the given samples and centers.
    Args
    X : ndarray of shape (n_samples, n_features) The input samples to assign to the labels. If sparse matrix, must
        be in CSR format.
    sample_weight : ndarray of shape (n_samples,)
        The weights for each observation in X.
    x_squared_norms : ndarray of shape (n_samples,)
        Precomputed squared euclidean norm of each data point, to speed up
        computations.
    centers : ndarray of shape (n_clusters, n_features)
        The cluster centers.
    n_threads : int, default=1
        The number of OpenMP threads to use for the computation. Parallelism is
        sample-wise on the main cython loop which assigns each sample to its
        closest center.
    return_inertia : bool, default=True
        Whether to compute and return the inertia.

    Returns:
    labels : ndarray of shape (n_samples,)
        The resulting assignment.
    inertia : float
        Sum of squared distances of samples to their closest cluster center.
        Inertia is only returned if return_inertia is True.
    """
    pairwise_distances = np.linalg.norm(
        X[:, None, :] - centers[None, :, :], axis=-1
    )  # (n_samples, n_clusters)

    labels = np.argmin(pairwise_distances, axis=1)

    return labels


class KMeansModel(object):
    """K-means Model class definition.
    
    Args:
    n_clusters: int, the number of cluters
    n_features: int, number of dimension for the cluster centroids
    init_ckpt: str, the initialization path of pretrained or initialized models.
    """
    def __init__(
        self,
        n_clusters: int,
        n_features: int,
        centroids: np.ndarray = None,
        n_threads: int = 1,
    ):
        self.n_clusters = n_clusters
        self.n_features = n_features
        if centroids is not None:
            self.centroids = centroids
        else:
            self.centroids = np.random.rand(n_clusters, n_features)
        self.n_threads = n_threads

    def load_ckpt(self, init_ckpt):
        assert os.path.exists(init_ckpt), f"{init_ckpt} does not exist."
        with open(init_ckpt, "rb") as f:
            ckpt = pickle.load(f)
        assert ckpt.n_clusters == self.n_clusters
        assert ckpt.n_features == self.n_features
        self.centroids = ckpt.centroids

    def predict(self, X, sample_weight=None):
        # labels = _labels_inertia_threadpool_limit(
        labels = _labels_inertia(
            X,
            sample_weight,
            self.centroids,
            n_threads=self.n_threads,
            return_inertia=False,
        )

        return labels
