import numpy as np
from sigmoid import *


def predict(theta1, theta2, x):
    # Useful values
    m = x.shape[0]
    num_labels = theta2.shape[0]

    # You need to return the following variable correctly
    p = np.zeros(m)

    # ===================== Your Code Here =====================
    # Instructions : Complete the following code to make predictions using
    #                your learned neural network. You should set p to a
    #                1-D array containing labels between 1 to num_labels.
    #
    a1 = x
    z2 = np.c_[np.zeros(m), a1] @ theta1.T
    a2 = sigmoid(z2)
    z3 = np.c_[np.zeros(a2.shape[0]), a2] @ theta2.T
    h = sigmoid(z3)
    p = np.argmax(h, axis=1) + 1
    return p


