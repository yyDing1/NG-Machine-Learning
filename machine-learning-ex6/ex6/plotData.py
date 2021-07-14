import matplotlib.pyplot as plt
import numpy as np


def plot_data(X, y):
    plt.figure()

    # ===================== Your Code Here =====================
    # Instructions : Plot the positive and negative examples on a
    #                2D plot, using the marker="+" for the positive
    #                examples and marker="o" for the negative examples
    #
    negative = X[y == 0]
    positive = X[y == 1]
    plt.scatter(negative[:, 0], negative[:, 1], marker='o')
    plt.scatter(positive[:, 0], positive[:, 1], marker='+')
