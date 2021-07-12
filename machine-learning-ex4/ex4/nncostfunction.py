import numpy as np
from sigmoid import *
from sklearn.preprocessing import OneHotEncoder
from sigmoidgradient import *


def nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmd):
    # Reshape nn_params back into the parameters theta1 and theta2, the weight 2-D arrays
    # for our two layer neural network
    theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, input_layer_size + 1)
    theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape(num_labels, hidden_layer_size + 1)

    # Useful value
    m = y.size
    encoder = OneHotEncoder(sparse=False)
    Y = encoder.fit_transform(y.reshape(m, 1))

    # You need to return the following variables correctly
    cost = 0
    theta1_grad = np.zeros(theta1.shape)  # 25 x 401
    theta2_grad = np.zeros(theta2.shape)  # 10 x 26

    # ===================== Your Code Here =====================
    # Instructions : You should complete the code by working thru the
    #                following parts
    #
    # Part 1 : Feedforward the neural network and return the cost in the
    #          variable cost. After implementing Part 1, you can verify that your
    #          cost function computation is correct by running ex4.py
    #
    # Part 2: Implement the backpropagation algorithm to compute the gradients
    #         theta1_grad and theta2_grad. You should return the partial derivatives of
    #         the cost function with respect to theta1 and theta2 in theta1_grad and
    #         theta2_grad, respectively. After implementing Part 2, you can check
    #         that your implementation is correct by running checkNNGradients
    #
    #         Note: The vector y passed into the function is a vector of labels
    #               containing values from 1..K. You need to map this vector into a 
    #               binary vector of 1's and 0's to be used with the neural network
    #               cost function.
    #
    #         Hint: We recommend implementing backpropagation using a for-loop
    #               over the training examples if you are implementing it for the 
    #               first time.
    #
    # Part 3: Implement regularization with the cost function and gradients.
    #
    #         Hint: You can implement this around the code for
    #               backpropagation. That is, you can compute the gradients for
    #               the regularization separately and then add them to theta1_grad
    #               and theta2_grad from Part 2.
    #
    a1 = X  # (5000, 400)
    z2 = np.c_[np.ones(m), a1] @ theta1.T  # (5000, 25)
    a2 = sigmoid(z2)  # (5000, 25)
    z3 = np.c_[np.ones(a2.shape[0]), a2] @ theta2.T  # (5000, 10)
    h = sigmoid(z3)  # (5000, 10)

    loss = 1 / m * np.sum(-Y * np.log(h) - (1 - Y) * np.log(1 - h))
    reg = lmd / (2 * X.shape[0]) * (np.sum(theta1[:, 1:] * theta1[:, 1:]) + np.sum(theta2[:, 1:] * theta2[:, 1:]))
    cost += loss + reg

    d3 = h - Y  # (5000, 10)
    d2 = d3 @ theta2[:, 1:] * sigmoid_gradient(z2)  # (5000, 25)
    theta2_grad += (d3.T @ np.c_[np.ones(a2.shape[0]), a2]) / m  # (10, 26)
    theta1_grad += (d2.T @ np.c_[np.ones(m), a1]) / m  # (25, 401)
    theta1_grad[:, 1:] += lmd / X.shape[0] * theta1[:, 1:]  # (10, 26)
    theta2_grad[:, 1:] += lmd / X.shape[0] * theta2[:, 1:]  # (25, 401)

    # ====================================================================================
    # Unroll gradients
    grad = np.concatenate([theta1_grad.flatten(), theta2_grad.flatten()])

    return cost, grad
