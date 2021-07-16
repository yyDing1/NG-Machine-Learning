import numpy as np


def kmeans_init_centroids(X, K):

    # You should return this value correctly
    (m, n) = X.shape
    centroids = np.zeros((K, n))

    # ===================== Your Code Here =====================
    # Instructions: You should set centroids to randomly chosen examples from
    #               the dataset X
    #
    choice = np.random.randint(0, m, K)
    centroids = X[choice, :]

    # ==========================================================

    return centroids
