import numpy as np


def normal_eqn(X, y):
    theta = np.linalg.inv(X.T @ X) @ X.T @ y

    # ===================== Your Code Here =====================
    # Instructions : Complete the code to compute the closed form solution
    #                to linear regression and put the result in theta
    #

    return theta
