import matplotlib.pyplot as plt
import numpy as np


def plot_data(X, y):
    plt.figure()

    # ===================== Your Code Here =====================
    # Instructions : Plot the positive and negative examples on a
    #                2D plot, using the marker="+" for the positive
    #                examples and marker="o" for the negative examples
    #
    negative = []
    positive = []
    for i in range(y.shape[0]):
        if y[i] == 0:
            negative.append(X[i])
        else:
            positive.append(X[i])
    negative = np.array(negative)
    positive = np.array(positive)
    plt.scatter(negative[:, 0], negative[:, 1], marker='o', c='r')
    plt.scatter(positive[:, 0], positive[:, 1], marker='+', c='b')
