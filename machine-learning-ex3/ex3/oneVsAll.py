import numpy as np
import scipy.optimize as opt
import lrCostFunction as lCF
from sigmoid import *


def one_vs_all(X, y, num_labels, lmd):
    # Some useful variables
    (m, n) = X.shape

    # You need to return the following variables correctly
    all_theta = np.zeros((num_labels, n + 1))

    # Add ones to the X data 2D-array
    X = np.c_[np.ones(m), X]

    for i in range(num_labels):
        print('Optimizing for handwritten number {}...'.format(i))
        # ===================== Your Code Here =====================
        # Instructions : You should complete the following code to train num_labels
        #                logistic regression classifiers with regularization
        #                parameter lambda
        #
        #
        # Hint: you can use y == c to obtain a vector of True(1)'s and False(0)'s that tell you
        #       whether the ground truth is true/false for this class
        #
        # Note: For this assignment, we recommend using opt.fmin_cg to optimize the cost
        #       function. It is okay to use a for-loop (for c in range(num_labels) to
        #       loop over the different classes
        #

        y_temp = np.zeros(m)
        for j in range(m):
            if y[j] == i + 1:
                y_temp[j] = 1
            else:
                y_temp[j] = 0
        init_theta = np.zeros(X.shape[1])

        def cost_func(t):
            return lCF.lr_cost_function(t, X, y_temp, lmd)[0]

        def grad_func(t):
            return lCF.lr_cost_function(t, X, y_temp, lmd)[1]

        # all_theta[i] = opt.fmin_cg(f=cost_func, fprime=grad_func, x0=init_theta, maxiter=400)
        all_theta[i] = opt.minimize(fun=cost_func, x0=init_theta, method='TNC', jac=grad_func).x
        # ============================================================    
        print('Done')

    return all_theta
