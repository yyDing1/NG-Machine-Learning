import numpy as np


def compute_cost(X, y, theta):
    # Initialize some useful values
    m = y.size
    cost = 0

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta.
    #                You should set the variable "cost" to the correct value.

    cost = 1 / (2 * X.shape[0]) * (X @ theta - y) @ (X @ theta - y)

    # ==========================================================

    return cost
