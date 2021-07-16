import numpy as np


def find_closest_centroids(X, centroids):
    # Set K
    K = centroids.shape[0]

    m = X.shape[0]

    # You need to return the following variables correctly.
    idx = np.zeros(m, dtype=np.int64)

    # ===================== Your Code Here =====================
    # Instructions : Go over every example, find its closest centroid, and store
    #                the index inside idx at the appropriate location.
    #                Concretely, idx[i] should contain the index of the centroid
    #                closest to example i. Hence, it should be a value in the
    #                range 0..k
    #
    for i in range(m):
        d = np.zeros(K)
        for j in range(K):
            d[j] = (X[i] - centroids[j]) @ (X[i] - centroids[j])
        idx[i] = np.argmin(d)

    # ==========================================================

    return idx
