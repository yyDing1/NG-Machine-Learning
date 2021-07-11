import numpy as np
from sigmoid import *


def cost_function_reg(theta, X, y, lmd):
    m = y.size

    # You need to return the following values correctly
    cost = 0
    grad = np.zeros(theta.shape)

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta
    #                You should set cost and grad correctly.
    #
    loss = -(y @ np.log(sigmoid(X @ theta)) + (1 - y) @ np.log(1 - sigmoid(X @ theta))) / m
    reg = lmd / (2 * m) * theta @ theta
    cost = loss + reg

    grad += 1 / m * X.T @ (sigmoid(X @ theta) - y) + lmd / m * theta

    # ===========================================================

    return cost, grad
