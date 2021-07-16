import numpy as np


def compute_centroids(X, idx, K):
    # Useful values
    (m, n) = X.shape

    # You need to return the following variable correctly.
    centroids = np.zeros((K, n))

    # ===================== Your Code Here =====================
    # Instructions: Go over every centroid and compute mean of all points that
    #               belong to it. Concretely, the row vector centroids[i]
    #               should contain the mean of the data points assigned to
    #               centroid i.
    #
    cnt = np.zeros(K)
    for i in range(m):
        centroids[idx[i]] += X[i]
        cnt[idx[i]] += 1
    for i in range(K):
        if cnt[i] == 0:
            continue
        centroids[i] /= cnt[i]

    # ==========================================================

    return centroids
