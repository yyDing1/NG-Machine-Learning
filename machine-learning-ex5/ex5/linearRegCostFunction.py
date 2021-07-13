import numpy as np


def linear_reg_cost_function(theta, x, y, lmd):
    # Initialize some useful values
    m = y.size

    # You need to return the following variables correctly
    cost = 0
    grad = np.zeros(theta.shape)

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost and gradient of regularized linear
    #                regression for a particular choice of theta
    #
    #                You should set 'cost' to the cost and 'grad'
    #                to the gradient
    #
    cost += 1 / (2 * m) * (x @ theta - y) @ (x @ theta - y) + lmd / (2 * m) * theta[1:] @ theta[1:]
    grad += 1 / m * (x.T @ x @ theta - x.T @ y) + lmd / m * np.insert(theta[1:], 0, 0)

    # ==========================================================

    return cost, grad
